.PHONY: all clean

all:
	mkdir -p forward_model/build
	$(MAKE) -C forward_model/build -f ../Makefile

clean:
	if test -d forward_model/build; then $(MAKE) -C forward_model/build -f ../Makefile clean; fi
