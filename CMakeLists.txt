cmake_minimum_required(VERSION 3.18)
project(
  brewster
  VERSION 2.0.0
  LANGUAGES C Fortran
)

# DISORT is trying to use a scalar as a rank-1 array
# Technically this should be fixed in DISORT, but this is a quick fix
set(CMAKE_Fortran_FLAGS
  "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch"
)

# Grab Python, 3.8 or newer
find_package(Python 3.8 REQUIRED
  COMPONENTS Interpreter Development.Module NumPy
)

# Grab the variables from a local Python installation
# F2PY headers
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Print out the discovered paths
include(CMakePrintHelpers)
cmake_print_variables(Python_INCLUDE_DIRS)
cmake_print_variables(F2PY_INCLUDE_DIR)
cmake_print_variables(Python_NumPy_INCLUDE_DIRS)

set(BREWSTER_SRC_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/src/fortran"
)

function(add_f2py_module name compile_flags link_marvin)
  set(src_files ${ARGN})
  set(output_name "${CMAKE_CURRENT_BINARY_DIR}/${name}.${Python_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}")

  # F2PY doesn't like spaces in filenames, so copy them to the build directory
  # and use relative paths
  foreach (src_file IN LISTS src_files)
    get_filename_component(src_file_name ${src_file} NAME)
    configure_file(${src_file}
      "${CMAKE_CURRENT_BINARY_DIR}"
      COPYONLY
    )
    list(APPEND src_files_relative ${src_file_name})
  endforeach()

  # FIXME
  # Use this contorted method to build the object rather than
  # add_python_library because the code doesn't work when using
  # the documented build without forcing f2py to call the compiler
  add_custom_command(
    TARGET marvin POST_BUILD
    BYPRODUCTS ${output_name} ${name}.pyf
    COMMAND ${Python_EXECUTABLE} -m "numpy.f2py"
            -m "${name}"
            -h "${name}.pyf"
            --overwrite-signature
             ${src_files_relative}
    COMMAND_EXPAND_LISTS
    COMMAND env LDFLAGS=-Wl,-rpath='\$$ORIGIN' NPY_DISTUTILS_APPEND_FLAGS=1
            ${Python_EXECUTABLE} -m numpy.f2py
            --fcompiler=gfortran
            --f90flags=\""${compile_flags}"\"
            -c "${name}.pyf"
            $<TARGET_FILE_NAME:marvin>
            ${src_files_relative}
    COMMAND_EXPAND_LISTS
    DEPENDS ${src_files}
  )
endfunction()

file(GLOB MARVIN_SOURCE_F90 src/fortran/*.f90)
file(GLOB MARVIN_SOURCE_F77 src/fortran/*.f)
add_library(marvin_f77 OBJECT ${MARVIN_SOURCE_F77})
set_target_properties(marvin_f77 PROPERTIES
  COMPILE_FLAGS "-fbounds-check -frecord-marker=4 -fdefault-real-8 -fdefault-double-8 -std=legacy"
  POSITION_INDEPENDENT_CODE ON
)
add_library(marvin_f90 OBJECT ${MARVIN_SOURCE_F90})
set_target_properties(marvin_f90 PROPERTIES
  COMPILE_FLAGS "-ffree-line-length-none -frecord-marker=4 -fbounds-check"
  POSITION_INDEPENDENT_CODE ON
)

add_library(marvin SHARED
  $<TARGET_OBJECTS:marvin_f77>
  $<TARGET_OBJECTS:marvin_f90>
)

add_f2py_module(forwardmodel "" ON
  ${BREWSTER_SRC_DIR}/sizes_mod.f90
  ${BREWSTER_SRC_DIR}/marv.f90
)

add_f2py_module(ciamod "" OFF
  ${BREWSTER_SRC_DIR}/sizes_mod.f90
  ${BREWSTER_SRC_DIR}/read_cia.f90
)

add_f2py_module(bbconv "" OFF
  ${BREWSTER_SRC_DIR}/bbconv.f90
)

add_f2py_module(cloudpost "-frecord-marker=4 -fbounds-check" OFF
  ${BREWSTER_SRC_DIR}/cloudpost.f90
)

install(TARGETS marvin DESTINATION brewster)
# Can't install output of custom commands, have to install the output
install(PROGRAMS
  "${CMAKE_CURRENT_BINARY_DIR}/bbconv.${Python_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}"
  "${CMAKE_CURRENT_BINARY_DIR}/ciamod.${Python_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}"
  "${CMAKE_CURRENT_BINARY_DIR}/cloudpost.${Python_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}"
  "${CMAKE_CURRENT_BINARY_DIR}/forwardmodel.${Python_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}"
  DESTINATION brewster
)
