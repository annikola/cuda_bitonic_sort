#ifndef SORTING_HELPERS_H
#define SORTING_HELPERS_H

int isAscending(int *A, int n);

int isAscending(int *A, int n) {

    int i;

    for (i = 0; i < n - 1; i++) {
        if (A[i] > A[i + 1]) {
            return 0;
        }
    }

    return 1;
}

#endif // SORTING_HELPERS_H