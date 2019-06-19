DIR_INC=./include
DIR_SRC=./src
DIR_BIN=./bin
DIR_OBJ=./obj

SRC=$(wildcard $(DIR_SRC)/*.cc)
OBJ=${patsubst %.c,${DIR_LIB}/%.o,$(notdir ${SRC})}

TARGET=main

BIN_TARGET=${DIR_BIN}/${TARGET}

CC=g++
CFLAGS = -g -Wall -I${DIR_INC}

${BIN_TARGET}:${OBJ}
	${CC} -o $@ ${OBJ}

${OBJ}/%.o:${SRC}/%.c
	$(CC) $(CFLAGS) -o $@ -c $<

.PHONY:clean

clean:
	$(RM) *.o $(target)


