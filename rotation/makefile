CFLAGS     = -g -MMD
CPPFLAGS   =
LIBFILES   =
TARGET     = ex_matvec
OBJFILES   = ex_matvec.o ex_utils.o
CLEANFILES = $(TARGET)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

all: $(TARGET)

$(TARGET) : $(OBJFILES)
	${CLINKER} -o $(TARGET) $(OBJFILES) ${PETSC_KSP_LIB}
