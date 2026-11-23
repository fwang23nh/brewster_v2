# Invoked by the repository Makefile from forward_model/build.
FC = gfortran
PYTHON = python
F2PY = $(PYTHON) -m numpy.f2py
F2PY_BACKEND = meson

FCFLAGS = -O2 -fPIC -ffree-line-length-none -frecord-marker=4 -fbounds-check -I/usr/include
F77FLAGS = -O2 -fPIC -fbounds-check -frecord-marker=4 -fdefault-real-8 -fdefault-double-8 -std=legacy -I/usr/include

# Meson compiles the wrappers in a temporary directory. Use absolute paths
# for the precompiled Fortran modules and static library.
F2PY_INCFLAGS = -I$(CURDIR) $(if $(wildcard /usr/include),-I/usr/include)
F2PY_LIBFLAGS = $(if $(wildcard /usr/local/lib),-L/usr/local/lib)
# Ninja launched by Meson cannot access Make's jobserver file descriptors.
F2PY_COMPILE = MAKEFLAGS= FC="$(FC)" $(F2PY) --backend $(F2PY_BACKEND) $(F2PY_INCFLAGS) $(F2PY_LIBFLAGS)
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

F77_OBJECTS = BDREF.o DISORT.o ErrPack.o LINPAK.o RDI1MACH.o bits_for_disort_f77.o \
              dsolver.o dtridgl.o gfluxi.o locate.o tristan.o
MODEL_OBJECTS = sizes_mod.o phys_const_mod.o define_types_mod.o common_arrays_mod.o \
                atmos_ops_mod.o gas_opacity_mod.o clouds_mod.o setup_RT_mod.o main_mod.o
EXTENSIONS = forwardmodel ciamod bbconv cloudpost

EXTENSION_FILES = $(addsuffix $(EXT_SUFFIX),$(EXTENSIONS))

.PHONY: all diagnostics clean
all: diagnostics $(EXTENSION_FILES)
	@printf '\nBrewster extensions ready in %s\n' "$(CURDIR)"

# Report the active toolchain even when every extension is already up to date.
# A version probe can fail; the build recipes still report dependency failures.
diagnostics:
	@printf '\nBrewster build diagnostics\n'
	@printf 'Platform: ' && uname -srm
	@printf 'Build directory: %s\nGNU Make: %s\nf2py backend: %s\n' "$(CURDIR)" "$(MAKE_VERSION)" "$(F2PY_BACKEND)"
	@printf 'Python command: %s\n' "$(PYTHON)"
	@$(PYTHON) -c 'import sys; print("Python:", sys.version.replace("\n", " ")); print("Interpreter:", sys.executable)' || printf 'Python version probe failed\n'
	@$(PYTHON) -c 'import numpy; print("NumPy:", numpy.__version__)' || printf 'NumPy version probe failed\n'
	@printf 'f2py: ' && ($(F2PY) -v || printf 'version probe failed\n')
	@printf 'Fortran compiler: %s\n' "$(FC)"
	@$(FC) --version || printf 'Fortran compiler version probe failed\n'
	@printf 'Meson: ' && (meson --version || printf 'version probe failed\n')
	@printf 'Ninja: ' && (ninja --version || printf 'version probe failed\n')
	@printf '\n'

$(F77_OBJECTS) $(MODEL_OBJECTS) $(addsuffix .pyf,$(EXTENSIONS)) $(EXTENSION_FILES): | diagnostics

.DELETE_ON_ERROR:

%.o: ../%.f90
	$(FC) $(FCFLAGS) -c $<

%.o: ../%.f
	$(FC) $(F77FLAGS) -c $<

define_types_mod.o: sizes_mod.o
common_arrays_mod.o: sizes_mod.o define_types_mod.o
atmos_ops_mod.o: sizes_mod.o common_arrays_mod.o phys_const_mod.o define_types_mod.o
gas_opacity_mod.o: sizes_mod.o common_arrays_mod.o phys_const_mod.o define_types_mod.o atmos_ops_mod.o tristan.o
clouds_mod.o: sizes_mod.o common_arrays_mod.o define_types_mod.o phys_const_mod.o
setup_RT_mod.o: sizes_mod.o common_arrays_mod.o define_types_mod.o phys_const_mod.o atmos_ops_mod.o bits_for_disort_f77.o DISORT.o dsolver.o gfluxi.o
main_mod.o: sizes_mod.o common_arrays_mod.o define_types_mod.o phys_const_mod.o atmos_ops_mod.o gas_opacity_mod.o clouds_mod.o setup_RT_mod.o

libmarvin.a: $(F77_OBJECTS) $(MODEL_OBJECTS)
	$(AR) rcs $@ $^

forwardmodel.pyf: ../sizes_mod.f90 ../marv.f90
	$(F2PY) -m forwardmodel -h $@ --overwrite-signature $^

ciamod.pyf: ../sizes_mod.f90 ../read_cia.f90
	$(F2PY) -m ciamod -h $@ --overwrite-signature $^

bbconv.pyf: ../bbconv.f90
	$(F2PY) -m bbconv -h $@ --overwrite-signature $^

cloudpost.pyf: ../cloudpost.f90
	$(F2PY) -m cloudpost -h $@ --overwrite-signature $^

forwardmodel$(EXT_SUFFIX): forwardmodel.pyf ../marv.f90 libmarvin.a
	$(F2PY_COMPILE) --f90flags="-O2" -L$(CURDIR) -lmarvin -c $(filter-out libmarvin.a,$^)

ciamod$(EXT_SUFFIX): ciamod.pyf ../sizes_mod.f90 ../read_cia.f90
	$(F2PY_COMPILE) --f90flags="-O2" -c $^

bbconv$(EXT_SUFFIX): bbconv.pyf ../bbconv.f90
	$(F2PY_COMPILE) --f90flags="-O2" -c $^

cloudpost$(EXT_SUFFIX): cloudpost.pyf ../cloudpost.f90
	$(F2PY_COMPILE) --f90flags="-O2 -fPIC -frecord-marker=4 -fbounds-check" -c $^

clean:
	rm -f *.o *.mod *.MOD *.pyf *.so *.a
