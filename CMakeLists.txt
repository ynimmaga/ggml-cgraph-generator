# CMake build for the ggml-cgraph dumpers. Replaces the original g++ build.sh (kept as a thin
# wrapper) with a build that works unchanged on Linux and Windows (MSVC).

# llama.cpp (the one hard dependency) is consumed in two modes, chosen at configure time:
#   PREBUILT -- <LLAMA_CPP_DIR>/build contains llama.cpp libraries; they are linked directly.
#               Fast, but requires llama.cpp to be built already, and this project must then
#               be built in the same configuration (e.g. Release) the tree was built in.
#   FALLBACK -- no prebuilt libraries found: llama.cpp is configured here as a subproject
#               and only the targets these tools need are built (first build: several minutes).
#
# Prebuilt llama.cpp build trees come in three layouts (llama.cpp puts shared libs in
# build/bin[/<Config>] and archive/import libs in per-target dirs; on Windows the "lib"
# prefix is stripped):
#   A. Visual Studio multi-config shared:  DLLs build/bin/<Config>/*.dll,
#      import libs build/src/<Config>/llama.lib,
#      build/ggml/src/<Config>/{ggml,ggml-base,ggml-cpu}.lib,
#      build/tools/mtmd/<Config>/mtmd.lib
#   B. Linux single-config shared:  build/bin/lib*.so
#   C. Linux single-config static:  build/src/libllama.a, build/ggml/src/lib*.a,
#      build/tools/mtmd/libmtmd.a
# (Windows Ninja/MSVC and MinGW single-config use the same dirs without the <Config> part.)

cmake_minimum_required(VERSION 3.16)
project(ggml-cgraph-generator LANGUAGES C CXX)   # C: the fallback builds ggml-cpu's .c files

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Match the old build.sh's -O2 on single-config generators (they default to no build type).
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(LLAMA_CPP_DIR   "../llama.cpp" CACHE PATH "llama.cpp source tree: prebuilt libs under <dir>/build are linked if present, else the tree is built from source")
set(WHISPER_CPP_DIR ""             CACHE PATH "built whisper.cpp source tree (optional; enables dump_whisper, prebuilt-only)")

# Resolve relative cache paths against the SOURCE dir (not cwd), like build.sh's default did.
foreach(_v LLAMA_CPP_DIR WHISPER_CPP_DIR)
    if(${_v} AND NOT IS_ABSOLUTE "${${_v}}")
        get_filename_component(${_v} "${${_v}}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endforeach()

if(MSVC)
    # /utf-8: the whisper unity TU and llama.cpp headers contain UTF-8 source text
    # /bigobj: dump_whisper compiles all of whisper.cpp as one TU; exceeds MSVC's section limit
    # _CRT_SECURE_NO_WARNINGS: fopen (cgraph_capture.hpp) / getenv (dump_cgraph.cpp) trip C4996
    # NOMINMAX: <windows.h> may be pulled in by mtmd/whisper sources compiled in our TUs
    add_compile_options(/utf-8 /bigobj)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

find_package(Threads REQUIRED)   # static llama.cpp needs pthreads; dump_whisper linked it directly

# Probe one library across the prebuilt layouts. Result vars: <out>_FLAT (no <Config> part:
# Linux shared/static, Windows single-config), <out>_Release, <out>_Debug. The CACHE entry is
# cleared first so a changed LLAMA_CPP_DIR/WHISPER_CPP_DIR never returns a stale result.
function(cgraph_probe_lib out name)
    set(_dirs ${ARGN})
    unset(${out}_FLAT    CACHE)
    unset(${out}_Release CACHE)
    unset(${out}_Debug   CACHE)
    foreach(_cfg Release Debug)
        find_library(${out}_${_cfg} NAMES ${name} PATHS ${_dirs} PATH_SUFFIXES ${_cfg} NO_DEFAULT_PATH)
        set(${out}_${_cfg} "${${out}_${_cfg}}" PARENT_SCOPE)
    endforeach()
    find_library(${out}_FLAT NAMES ${name} PATHS ${_dirs} NO_DEFAULT_PATH)
    set(${out}_FLAT "${${out}_FLAT}" PARENT_SCOPE)
endfunction()

# Create an imported target for one probed library. Shared targets on Windows get the
# IMPORTED_IMPLIB (link against) + DLL (runtime/copy) pair; static targets get a plain
# location. Pass OPTIONAL as the 7th argument to create nothing when no location was found.
function(cgraph_import_lib target base dllname includes shared_flag bin_dir)
    set(_optional OFF)
    if(ARGC GREATER 6 AND ARGV6 STREQUAL "OPTIONAL")
        set(_optional ON)
    endif()

    set(_any OFF)
    if(NOT ${base}_FLAT AND NOT ${base}_Release AND NOT ${base}_Debug)
        if(_optional)
            return()
        endif()
        message(FATAL_ERROR "internal error: no usable library location found for ${target}")
    endif()

    if(${shared_flag})
        add_library(${target} SHARED IMPORTED)
    else()
        add_library(${target} STATIC IMPORTED)
    endif()
    set_target_properties(${target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${includes}")

    # Per-config locations. NOTE the UPPERCASE config suffix: CMake's location lookup
    # uppercases the configuration name before building the property name (cmTarget.cxx), so
    # IMPORTED_IMPLIB_RELEASE matches but a camel-case _Release suffix would not. Both
    # casings are set for compatibility with other CMake versions' lookup order.
    foreach(_cfg Release Debug)
        if(${base}_${_cfg})
            set(_any ON)
            list(APPEND _configs ${_cfg})
            string(TOUPPER ${_cfg} _cfgU)
            if(${shared_flag} AND WIN32)
                # Multi-config: import lib in the per-target dir, DLL in bin/<Config>.
                set_target_properties(${target} PROPERTIES
                    IMPORTED_IMPLIB_${_cfgU}   "${${base}_${_cfg}}"
                    IMPORTED_IMPLIB_${_cfg}    "${${base}_${_cfg}}"
                    IMPORTED_LOCATION_${_cfgU} "${bin_dir}/${_cfg}/${dllname}.dll"
                    IMPORTED_LOCATION_${_cfg}  "${bin_dir}/${_cfg}/${dllname}.dll")
            else()
                set_target_properties(${target} PROPERTIES
                    IMPORTED_LOCATION_${_cfgU} "${${base}_${_cfg}}"
                    IMPORTED_LOCATION_${_cfg}  "${${base}_${_cfg}}")
            endif()
        endif()
    endforeach()
    if(_configs)
        # Restrict to the configurations that actually have locations, so generation does not
        # probe (and warn about) configs this tree was never built in.
        set_target_properties(${target} PROPERTIES IMPORTED_CONFIGURATIONS "${_configs}")
    endif()
    if(${base}_FLAT)
        set(_any ON)
        if(${shared_flag} AND WIN32)
            if(${base}_FLAT MATCHES "\\.dll\\.a$")   # MinGW: lib<name>.dll.a + lib<name>.dll
                set_target_properties(${target} PROPERTIES
                    IMPORTED_IMPLIB   "${${base}_FLAT}"
                    IMPORTED_LOCATION "${bin_dir}/lib${dllname}.dll")
            else()                                  # MSVC single-config: <name>.lib + <name>.dll
                set_target_properties(${target} PROPERTIES
                    IMPORTED_IMPLIB   "${${base}_FLAT}"
                    IMPORTED_LOCATION "${bin_dir}/${dllname}.dll")
            endif()
        else()
            set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${${base}_FLAT}")
        endif()
    endif()
endfunction()

# Copy the runtime DLLs of the given imported/real targets next to <exe> (POST_BUILD).
function(cgraph_copy_runtime_dlls exe shared_flag)
    if(NOT (WIN32 AND ${shared_flag}))
        return()
    endif()
    set(_dlls)
    foreach(_lib IN LISTS ARGN)
        if(TARGET ${_lib})
            list(APPEND _dlls "$<TARGET_FILE:${_lib}>")
        endif()
    endforeach()
    if(NOT _dlls)
        return()
    endif()
    add_custom_command(TARGET ${exe} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_dlls} "$<TARGET_FILE_DIR:${exe}>"
        COMMENT "Copying runtime DLLs next to ${exe}")
endfunction()

set(CGRAPH_LLAMA_PREBUILT OFF)
set(CGRAPH_LLAMA_SHARED   OFF)
set(CGRAPH_HAVE_MTMD      OFF)
set(CGRAPH_LLAMA_BIN_DIR  "")

# ---------------------------------------------------------------------------------------
# Prebuilt mode: probe the llama.cpp build tree.
# ---------------------------------------------------------------------------------------
if(EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt" AND EXISTS "${LLAMA_CPP_DIR}/build")
    set(_LB "${LLAMA_CPP_DIR}/build")

    # Import libs / static libs live in per-target dirs; shared libs (and MinGW import libs)
    # live in bin/. Probe both.
    cgraph_probe_lib(LLAMA_LIB     llama     "${_LB}/src"        "${_LB}/bin")
    cgraph_probe_lib(GGML_LIB      ggml      "${_LB}/ggml/src"   "${_LB}/bin")
    cgraph_probe_lib(GGML_BASE_LIB ggml-base "${_LB}/ggml/src"   "${_LB}/bin")
    cgraph_probe_lib(GGML_CPU_LIB  ggml-cpu  "${_LB}/ggml/src"   "${_LB}/bin")
    cgraph_probe_lib(MTMD_LIB      mtmd      "${_LB}/tools/mtmd" "${_LB}/bin")

    # Valid prebuilt mode needs the three core libs in at least one location.
    set(_core_ok ON)
    foreach(_b LLAMA_LIB GGML_LIB GGML_BASE_LIB)
        if(NOT ${_b}_FLAT AND NOT ${_b}_Release AND NOT ${_b}_Debug)
            set(_core_ok OFF)
        endif()
    endforeach()
    if(_core_ok)
        set(CGRAPH_LLAMA_PREBUILT ON)
        set(CGRAPH_LLAMA_BIN_DIR "${_LB}/bin")
        if(MTMD_LIB_FLAT OR MTMD_LIB_Release OR MTMD_LIB_Debug)
            set(CGRAPH_HAVE_MTMD ON)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------------------
# Shared-vs-static (drives dllimport defines, DLL copying, static-only link extras).
# ---------------------------------------------------------------------------------------
if(CGRAPH_LLAMA_PREBUILT)
    foreach(_p "${CGRAPH_LLAMA_BIN_DIR}/Release/llama.dll" "${CGRAPH_LLAMA_BIN_DIR}/Debug/llama.dll"
               "${CGRAPH_LLAMA_BIN_DIR}/llama.dll"         "${CGRAPH_LLAMA_BIN_DIR}/libllama.dll")
        if(EXISTS "${_p}")
            set(CGRAPH_LLAMA_SHARED ON)
        endif()
    endforeach()
    if(LLAMA_LIB_FLAT MATCHES "\\.(so|dylib|dll\\.a)$")
        set(CGRAPH_LLAMA_SHARED ON)
    endif()
endif()

# ---------------------------------------------------------------------------------------
# Imported targets (prebuilt) or aliases to real targets (fallback). The executables below
# only ever see llama_cpp::* names.
# ---------------------------------------------------------------------------------------
if(CGRAPH_LLAMA_PREBUILT)
    # ggml-cpu is a hard requirement for static linking and for the Windows DLL copy
    # (ggml.dll imports ggml-cpu.dll), but is not on the link line for shared libs.
    if(CGRAPH_LLAMA_SHARED)
        if(WIN32 AND NOT GGML_CPU_LIB_FLAT AND NOT GGML_CPU_LIB_Release AND NOT GGML_CPU_LIB_Debug)
            message(WARNING "ggml-cpu library not found in the prebuilt tree; dump_cgraph and "
                            "dump_vision may fail to start (ggml.dll imports ggml-cpu.dll)")
        endif()
    else()
        if(NOT GGML_CPU_LIB_FLAT AND NOT GGML_CPU_LIB_Release AND NOT GGML_CPU_LIB_Debug)
            message(FATAL_ERROR "static llama.cpp build detected but the ggml-cpu library was "
                                "not found under ${_LB} (it must be on the link line)")
        endif()
    endif()

    cgraph_import_lib(llama_cpp::ggml-base GGML_BASE_LIB ggml-base "${LLAMA_CPP_DIR}/ggml/include"
        CGRAPH_LLAMA_SHARED "${CGRAPH_LLAMA_BIN_DIR}")
    cgraph_import_lib(llama_cpp::ggml      GGML_LIB      ggml      "${LLAMA_CPP_DIR}/ggml/include"
        CGRAPH_LLAMA_SHARED "${CGRAPH_LLAMA_BIN_DIR}")
    cgraph_import_lib(llama_cpp::ggml-cpu  GGML_CPU_LIB  ggml-cpu  "${LLAMA_CPP_DIR}/ggml/include"
        CGRAPH_LLAMA_SHARED "${CGRAPH_LLAMA_BIN_DIR}" OPTIONAL)
    cgraph_import_lib(llama_cpp::llama     LLAMA_LIB     llama
        "${LLAMA_CPP_DIR}/include;${LLAMA_CPP_DIR}/ggml/include"
        CGRAPH_LLAMA_SHARED "${CGRAPH_LLAMA_BIN_DIR}")
    if(CGRAPH_HAVE_MTMD)
        cgraph_import_lib(llama_cpp::mtmd  MTMD_LIB      mtmd
            "${LLAMA_CPP_DIR}/include;${LLAMA_CPP_DIR}/ggml/include;${LLAMA_CPP_DIR}/tools/mtmd"
            CGRAPH_LLAMA_SHARED "${CGRAPH_LLAMA_BIN_DIR}")
    endif()

    # Interface link structure mirrors llama.cpp's own: llama PUBLIC ggml, ggml PUBLIC
    # ggml-base (+ ggml-cpu, which is PUBLIC-linked via ggml), mtmd PUBLIC ggml llama.
    target_link_libraries(llama_cpp::ggml  INTERFACE llama_cpp::ggml-base)
    target_link_libraries(llama_cpp::llama INTERFACE llama_cpp::ggml)
    if(CGRAPH_HAVE_MTMD)
        target_link_libraries(llama_cpp::mtmd INTERFACE llama_cpp::llama llama_cpp::ggml)
    endif()

    if(CGRAPH_LLAMA_SHARED AND WIN32)
        # The DLLs were built with these shared defines; consumers need dllimport.
        target_compile_definitions(llama_cpp::llama     INTERFACE LLAMA_SHARED)
        target_compile_definitions(llama_cpp::ggml      INTERFACE GGML_SHARED)
        target_compile_definitions(llama_cpp::ggml-base INTERFACE GGML_SHARED)
        if(TARGET llama_cpp::ggml-cpu)
            target_compile_definitions(llama_cpp::ggml-cpu INTERFACE GGML_SHARED)
        endif()
    endif()

    if(NOT CGRAPH_LLAMA_SHARED)
        # Static: ggml-cpu is only PUBLIC-linked via ggml, so it must appear on the exe's link
        # line, and a static llama.cpp pulls in pthread/dl/m.
        target_link_libraries(llama_cpp::ggml INTERFACE llama_cpp::ggml-cpu)
        if(UNIX)
            target_link_libraries(llama_cpp::llama INTERFACE Threads::Threads ${CMAKE_DL_LIBS} m)
        endif()
    endif()

    set(_linkable)
    foreach(_cfg Release Debug)
        if(LLAMA_LIB_${_cfg})
            list(APPEND _linkable ${_cfg})
        endif()
    endforeach()
    if(LLAMA_LIB_FLAT)
        list(APPEND _linkable "<single-config>")
    endif()
    string(REPLACE ";" ", " _linkable "${_linkable}")
    set(_shared_state "static")
    if(CGRAPH_LLAMA_SHARED)
        set(_shared_state "shared")
    endif()
    message(STATUS "llama.cpp: prebuilt ${_shared_state} at ${LLAMA_CPP_DIR} "
                   "(linkable configurations: ${_linkable})")
else()
    if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "LLAMA_CPP_DIR=${LLAMA_CPP_DIR} is not a llama.cpp source tree "
                            "(no CMakeLists.txt), and no prebuilt libraries were found. Point "
                            "LLAMA_CPP_DIR at a llama.cpp checkout.")
    endif()
    message(STATUS "llama.cpp: no prebuilt libraries under ${LLAMA_CPP_DIR}/build; building "
                   "llama.cpp from source as a subproject (first build takes several minutes)")

    # Minimal build: only what these tools link. mtmd's library links just PUBLIC ggml llama
    # (no llama-common; upstream enforces that with a guard), and its CLI tools -- the only
    # common.h consumers -- are gated on LLAMA_BUILD_TOOLS, so tools/common can stay OFF.
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_UI       OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_MTMD     ON  CACHE BOOL "" FORCE)

    add_subdirectory("${LLAMA_CPP_DIR}" llama_cpp_src EXCLUDE_FROM_ALL)

    # EXCLUDE_FROM_ALL: only what these executables link gets built. The real targets carry
    # their include dirs and shared-lib defines themselves.
    add_library(llama_cpp::llama     ALIAS llama)
    add_library(llama_cpp::ggml      ALIAS ggml)
    add_library(llama_cpp::ggml-base ALIAS ggml-base)
    add_library(llama_cpp::ggml-cpu  ALIAS ggml-cpu)
    add_library(llama_cpp::mtmd      ALIAS mtmd)
    set(CGRAPH_HAVE_MTMD ON)

    # llama.cpp defaults BUILD_SHARED_LIBS to ON (except MinGW/Emscripten), so a fallback build
    # usually produces shared libs; honor whatever was configured for runtime handling.
    get_target_property(_llama_type llama TYPE)
    if(_llama_type STREQUAL "SHARED_LIBRARY")
        set(CGRAPH_LLAMA_SHARED ON)
    endif()

    # llama.cpp's own CMake hard-codes library output to ${CMAKE_BINARY_DIR}/bin -- OUR bin.
    set(CGRAPH_LLAMA_BIN_DIR "${CMAKE_BINARY_DIR}/bin")

    message(STATUS "llama.cpp: building from source (${_llama_type})")
endif()

# ---------------------------------------------------------------------------------------
# dump_cgraph / dump_vision
# ---------------------------------------------------------------------------------------
add_executable(dump_cgraph dump_cgraph.cpp)
target_include_directories(dump_cgraph PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")   # cgraph_capture.hpp
target_link_libraries(dump_cgraph PRIVATE llama_cpp::llama)
cgraph_copy_runtime_dlls(dump_cgraph CGRAPH_LLAMA_SHARED
    llama_cpp::llama llama_cpp::ggml llama_cpp::ggml-base llama_cpp::ggml-cpu)

if(CGRAPH_HAVE_MTMD)
    add_executable(dump_vision dump_vision.cpp)
    target_include_directories(dump_vision PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(dump_vision PRIVATE llama_cpp::mtmd)   # pulls llama+ggml transitively
    cgraph_copy_runtime_dlls(dump_vision CGRAPH_LLAMA_SHARED
        llama_cpp::llama llama_cpp::ggml llama_cpp::ggml-base llama_cpp::ggml-cpu llama_cpp::mtmd)
else()
    message(WARNING "mtmd library not found in the prebuilt llama.cpp tree; dump_vision will "
                    "NOT be built. Build llama.cpp with LLAMA_BUILD_TOOLS=ON (its default), or "
                    "remove its build/ directory to trigger a from-source build.")
endif()

if(UNIX AND NOT APPLE)
    set_target_properties(dump_cgraph PROPERTIES BUILD_RPATH "${CGRAPH_LLAMA_BIN_DIR}")
    if(TARGET dump_vision)
        set_target_properties(dump_vision PROPERTIES BUILD_RPATH "${CGRAPH_LLAMA_BIN_DIR}")
    endif()
endif()

# ---------------------------------------------------------------------------------------
# dump_whisper (optional). WHISPER_CPP_DIR must point at a BUILT whisper.cpp tree: whisper
# is linked prebuilt-only. Its vendored ggml build tree uses the same layouts as llama.cpp.
# ---------------------------------------------------------------------------------------
if(WHISPER_CPP_DIR)
    set(_W "${WHISPER_CPP_DIR}")

    if(NOT EXISTS "${_W}/include/whisper.h" OR NOT EXISTS "${_W}/src/whisper.cpp")
        message(FATAL_ERROR "WHISPER_CPP_DIR=${_W} is not a whisper.cpp source tree "
                            "(need include/whisper.h and src/whisper.cpp)")
    endif()
    if(NOT EXISTS "${_W}/src/aneforge/whisper-aneforge.cpp")
        message(FATAL_ERROR "whisper-aneforge.cpp not found at "
                            "${_W}/src/aneforge/whisper-aneforge.cpp -- dump_whisper needs it "
                            "(it registers the CPU backend the GGUF writer requires). Expected "
                            "a whisper.cpp tree with an aneforge directory.")
    endif()
    if(NOT EXISTS "${_W}/build")
        message(FATAL_ERROR "WHISPER_CPP_DIR=${_W} has no build/ directory. dump_whisper links "
                            "whisper.cpp prebuilt libraries only: build whisper.cpp first (its "
                            "ggml libraries land in <dir>/build/bin[/<Config>]).")
    endif()
    set(_WB "${_W}/build")

    cgraph_probe_lib(WHISPER_GGML_LIB      ggml      "${_WB}/ggml/src" "${_WB}/bin")
    cgraph_probe_lib(WHISPER_GGML_BASE_LIB ggml-base "${_WB}/ggml/src" "${_WB}/bin")
    cgraph_probe_lib(WHISPER_GGML_CPU_LIB  ggml-cpu  "${_WB}/ggml/src" "${_WB}/bin")

    foreach(_b WHISPER_GGML_LIB WHISPER_GGML_BASE_LIB WHISPER_GGML_CPU_LIB)
        if(NOT ${_b}_FLAT AND NOT ${_b}_Release AND NOT ${_b}_Debug)
            message(FATAL_ERROR "whisper.cpp ggml libraries not found under ${_WB} (looked in "
                                "bin/ and ggml/src/, with and without Release/Debug config "
                                "subdirs). dump_whisper links whisper.cpp prebuilt libraries "
                                "only.")
        endif()
    endforeach()

    set(CGRAPH_WHISPER_SHARED OFF)
    foreach(_p "${_WB}/bin/Release/ggml.dll" "${_WB}/bin/Debug/ggml.dll"
               "${_WB}/bin/ggml.dll"         "${_WB}/bin/libggml.dll")
        if(EXISTS "${_p}")
            set(CGRAPH_WHISPER_SHARED ON)
        endif()
    endforeach()
    if(WHISPER_GGML_LIB_FLAT MATCHES "\\.(so|dylib|dll\\.a)$")
        set(CGRAPH_WHISPER_SHARED ON)
    endif()
    set(CGRAPH_WHISPER_BIN_DIR "${_WB}/bin")

    cgraph_import_lib(whisper_cpp::ggml-base WHISPER_GGML_BASE_LIB ggml-base
        "${_W}/include;${_W}/ggml/include" CGRAPH_WHISPER_SHARED "${CGRAPH_WHISPER_BIN_DIR}")
    cgraph_import_lib(whisper_cpp::ggml WHISPER_GGML_LIB ggml
        "${_W}/include;${_W}/ggml/include" CGRAPH_WHISPER_SHARED "${CGRAPH_WHISPER_BIN_DIR}")
    cgraph_import_lib(whisper_cpp::ggml-cpu WHISPER_GGML_CPU_LIB ggml-cpu
        "${_W}/include;${_W}/ggml/include" CGRAPH_WHISPER_SHARED "${CGRAPH_WHISPER_BIN_DIR}")

    target_link_libraries(whisper_cpp::ggml INTERFACE whisper_cpp::ggml-base)
    if(CGRAPH_WHISPER_SHARED AND WIN32)
        target_compile_definitions(whisper_cpp::ggml      INTERFACE GGML_SHARED)
        target_compile_definitions(whisper_cpp::ggml-base INTERFACE GGML_SHARED)
        target_compile_definitions(whisper_cpp::ggml-cpu  INTERFACE GGML_SHARED)
    endif()
    if(NOT CGRAPH_WHISPER_SHARED)
        target_link_libraries(whisper_cpp::ggml INTERFACE whisper_cpp::ggml-cpu)
        if(UNIX)
            target_link_libraries(whisper_cpp::ggml INTERFACE Threads::Threads ${CMAKE_DL_LIBS} m)
        endif()
    endif()

    # WHISPER_VERSION: the old build.sh scraped it from build/build.ninja, which only exists
    # for Ninja/Makefiles builds. Read it from the header instead; fall back to "unknown" (the
    # old script's fallback) when the header carries no definition at all.
    set(_wver "")
    set(_wver_mentioned OFF)
    file(STRINGS "${_W}/include/whisper.h" _wver_lines REGEX "WHISPER_VERSION")
    foreach(_line ${_wver_lines})
        if(_line MATCHES "#define[ \t]+WHISPER_VERSION[ \t]+\"([^\"]*)\"")
            set(_wver "${CMAKE_MATCH_1}")
            break()
        endif()
        if(_line MATCHES "WHISPER_VERSION")
            set(_wver_mentioned ON)
        endif()
    endforeach()

    add_executable(dump_whisper dump_whisper.cpp "${_W}/src/aneforge/whisper-aneforge.cpp")
    target_include_directories(dump_whisper PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${_W}/include" "${_W}/src" "${_W}/ggml/include")
    if(_wver)
        target_compile_definitions(dump_whisper PRIVATE "WHISPER_VERSION=\"${_wver}\"")
    elseif(NOT _wver_mentioned)
        # The header mentions no WHISPER_VERSION at all: guarantee one, or the unity TU may
        # not compile.
        target_compile_definitions(dump_whisper PRIVATE "WHISPER_VERSION=\"unknown\"")
    endif()
    target_link_libraries(dump_whisper PRIVATE
        whisper_cpp::ggml whisper_cpp::ggml-base whisper_cpp::ggml-cpu Threads::Threads)

    # whisper.cpp vendors a DIFFERENT ggml fork whose DLLs share names with llama.cpp's
    # (ggml.dll, ggml-base.dll, ggml-cpu.dll). Keep this exe and its copied DLLs in their own
    # directory so dump_cgraph/dump_vision never load the wrong ggml.
    set_target_properties(dump_whisper PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/whisper")
    cgraph_copy_runtime_dlls(dump_whisper CGRAPH_WHISPER_SHARED
        whisper_cpp::ggml whisper_cpp::ggml-base whisper_cpp::ggml-cpu)
    if(UNIX AND NOT APPLE)
        set_target_properties(dump_whisper PROPERTIES BUILD_RPATH "${CGRAPH_WHISPER_BIN_DIR}")
    endif()
endif()
