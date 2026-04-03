CC     = clang
CFLAGS = -Wall -Wextra -Werror -O0 -g

numpee.a: vec.o
	ar rcs numpee.a vec.o

vec.s: vec.c
	$(CC) $(CFLAGS) -S vec.c -DNDEBUG

vec.o: vec.c
	$(CC) $(CFLAGS) -c vec.c

test: vec.c
	$(CC) $(CFLAGS) -DTEST -o test_vec vec.c && ./test_vec

clean:
	rm -f numpee.a vec.o vec.s test_vec
