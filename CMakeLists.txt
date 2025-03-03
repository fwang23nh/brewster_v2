cmake_minimum_required(VERSION 3.18)
project(
  brewster
  VERSION 2.0.0
  LANGUAGES C Fortran
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

set(BREWSTER_F77_FLAGS "${BREWSTER_F77_FLAGS} -frecord-marker=4 -fdefault-real-8 -fdefault-double-8 -std=legacy")
set(BREWSTER_F90_FLAGS "${BREWSTER_F90_FLAGS} -ffree-line-length-none -frecord-marker=4")

# Print out the discovered paths
include(CMakePrintHelpers)
cmake_print_variables(Python_INCLUDE_DIRS)
cmake_print_variables(F2PY_INCLUDE_DIR)
cmake_print_variables(Python_NumPy_INCLUDE_DIRS)

set(BREWSTER_SRC_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/src/fortran"
)

function(add_f2py_module name compile_flags)
  set(src_files ${ARGN})
  # gfortran always uses .so on mac and linux
  set(output_name "${CMAKE_CURRENT_BINARY_DIR}/${name}.${Python_SOABI}.so")

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

  if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(ldflags LDFLAGS=-Wl,-rpath='$$ORIGIN')
  endif()

  # FIXME
  # Use this contorted method to build the object rather than
  # add_python_library because the code doesn't work when using
  # the documented build without forcing f2py to call the compiler
  get_target_property(MARVIN_FFLAGS marvin_f90 COMPILE_OPTIONS)
  add_custom_command(
    TARGET marvin POST_BUILD
    BYPRODUCTS ${output_name} ${name}.pyf
    COMMAND ${Python_EXECUTABLE} -m "numpy.f2py"
            -m "${name}"
            -h "${name}.pyf"
            --overwrite-signature
             ${src_files_relative}
    COMMAND_EXPAND_LISTS
    COMMAND env ${ldflags} NPY_DISTUTILS_APPEND_FLAGS=1
            ${Python_EXECUTABLE} -m numpy.f2py
            --fcompiler=gfortran
            --f90flags=\""${compile_flags} ${BREWSTER_F90_FLAGS}"\"
            -c "${name}.pyf"
            $<TARGET_FILE_NAME:marvin>
    COMMAND_EXPAND_LISTS
    DEPENDS ${src_files}
  )
endfunction()

file(GLOB MARVIN_SOURCE_F90 src/fortran/*.f90)
file(GLOB MARVIN_SOURCE_F77 src/fortran/*.f)
add_library(marvin_f77 OBJECT ${MARVIN_SOURCE_F77})
set_target_properties(marvin_f77 PROPERTIES
  COMPILE_FLAGS "${BREWSTER_F77_FLAGS}"
  POSITION_INDEPENDENT_CODE ON
)
add_library(marvin_f90 OBJECT ${MARVIN_SOURCE_F90})
set_target_properties(marvin_f90 PROPERTIES
  COMPILE_FLAGS "${BREWSTER_F90_FLAGS}"
  POSITION_INDEPENDENT_CODE ON
)

add_library(marvin SHARED
  $<TARGET_OBJECTS:marvin_f77>
  $<TARGET_OBJECTS:marvin_f90>
)

add_f2py_module(forwardmodel ""
  ${BREWSTER_SRC_DIR}/sizes_mod.f90
  ${BREWSTER_SRC_DIR}/marv.f90
)

add_f2py_module(ciamod ""
  ${BREWSTER_SRC_DIR}/sizes_mod.f90
  ${BREWSTER_SRC_DIR}/read_cia.f90
)

add_f2py_module(bbconv ""
  ${BREWSTER_SRC_DIR}/bbconv.f90
)

add_f2py_module(cloudpost "-frecord-marker=4 -fbounds-check"
  ${BREWSTER_SRC_DIR}/cloudpost.f90
)

install(TARGETS marvin DESTINATION brewster)
# Can't install output of custom commands, have to install the output
install(PROGRAMS
  "${CMAKE_CURRENT_BINARY_DIR}/bbconv.${Python_SOABI}.so"
  "${CMAKE_CURRENT_BINARY_DIR}/ciamod.${Python_SOABI}.so"
  "${CMAKE_CURRENT_BINARY_DIR}/cloudpost.${Python_SOABI}.so"
  "${CMAKE_CURRENT_BINARY_DIR}/forwardmodel.${Python_SOABI}.so"
  DESTINATION brewster
)
