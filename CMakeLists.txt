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

function(add_f2py_module name)
  set(src_files ${ARGN})
  add_custom_target(
    ${name}_gen
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${name}module.c"
            "${CMAKE_CURRENT_BINARY_DIR}/${name}-f2pywrappers2.f90"
  )
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${name}module.c"
           "${CMAKE_CURRENT_BINARY_DIR}/${name}-f2pywrappers2.f90"
    COMMAND ${Python_EXECUTABLE} -m "numpy.f2py"
                     ${src_files}
                     -m "${name}"
                     --lower
    DEPENDS ${src_files}
  )

  Python_add_library(${name} MODULE WITH_SOABI
    "${CMAKE_CURRENT_BINARY_DIR}/${name}module.c"
    "${F2PY_INCLUDE_DIR}/fortranobject.c"
    "${name}-f2pywrappers2.f90"
  )

  target_link_libraries(${name} PRIVATE Python::NumPy marvin)
  add_dependencies(${name} ${name}_gen)
  target_include_directories(${name} PRIVATE "${F2PY_INCLUDE_DIR}")
  set_target_properties(${name} PROPERTIES
    INSTALL_RPATH "\$ORIGIN"
  )
endfunction()

add_f2py_module(bbconv
  ${BREWSTER_SRC_DIR}/bbconv.f90
)

add_f2py_module(ciamod
  ${BREWSTER_SRC_DIR}/sizes_mod.f90
  ${BREWSTER_SRC_DIR}/read_cia.f90
)

add_f2py_module(cloudpost
  ${BREWSTER_SRC_DIR}/cloudpost.f90
)

add_f2py_module(forwardmodel
  ${BREWSTER_SRC_DIR}/sizes_mod.f90
  ${BREWSTER_SRC_DIR}/marv.f90
)

file(GLOB MARVIN_SOURCE src/fortran/*.f*)
add_library(marvin SHARED ${MARVIN_SOURCE})

install(TARGETS marvin bbconv ciamod cloudpost forwardmodel DESTINATION brewster)
