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
  "${CMAKE_CURRENT_SOURCE_DIR}/brewster/fortran"
)

function(add_f2py_module name)
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

  add_custom_target(
    ${name}_gen
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${name}module.c"
            "${CMAKE_CURRENT_BINARY_DIR}/${name}-f2pywrappers2.f90"
  )
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${name}module.c"
           "${CMAKE_CURRENT_BINARY_DIR}/${name}-f2pywrappers2.f90"
    COMMAND ${Python_EXECUTABLE} -m "numpy.f2py"
            ${src_files_relative}
            -m "${name}"
            --lower
    COMMAND_EXPAND_LISTS
    DEPENDS ${src_files}
  )

  Python_add_library(${name} MODULE WITH_SOABI
    "${CMAKE_CURRENT_BINARY_DIR}/${name}module.c"
    "${F2PY_INCLUDE_DIR}/fortranobject.c"
    "${name}-f2pywrappers2.f90"
  )

  target_link_libraries(${name} PRIVATE Python::NumPy marvin)
  set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/${name}-f2pywrappers2.f90"
    COMPILE_OPTIONS "${BREWSTER_F90_FLAGS}"
  )
  add_dependencies(${name} ${name}_gen)
  target_include_directories(${name} PRIVATE "${F2PY_INCLUDE_DIR}")
  set_target_properties(${name} PROPERTIES
    INSTALL_RPATH "\$ORIGIN"
  )
endfunction()

file(GLOB MARVIN_SOURCE_F77 brewster/fortran/*.f)
file(GLOB MARVIN_SOURCE_F90 brewster/fortran/*.f90)
set_source_files_properties(${MARVIN_SOURCE_F77} PROPERTIES
  COMPILE_FLAGS "${BREWSTER_F77_FLAGS}"
)
set_source_files_properties(${MARVIN_SOURCE_F90} PROPERTIES
  COMPILE_FLAGS "${BREWSTER_F90_FLAGS}"
)
add_library(marvin SHARED
  ${MARVIN_SOURCE_F77}
  ${MARVIN_SOURCE_F90}
)

add_f2py_module(forwardmodel
  ${BREWSTER_SRC_DIR}/sizes_mod.f90
  ${BREWSTER_SRC_DIR}/marv.f90
)

add_f2py_module(ciamod
  ${BREWSTER_SRC_DIR}/sizes_mod.f90
  ${BREWSTER_SRC_DIR}/read_cia.f90
)

add_f2py_module(bbconv
  ${BREWSTER_SRC_DIR}/sizes_mod.f90
  ${BREWSTER_SRC_DIR}/bbconv.f90
)

add_f2py_module(cloudpost
  ${BREWSTER_SRC_DIR}/cloudpost.f90
)

install(TARGETS marvin forwardmodel ciamod bbconv cloudpost DESTINATION brewster)
