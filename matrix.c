#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    
    //TODO//
    
    if (rows < 1 || cols < 1) {
        PyErr_SetString(PyExc_TypeError, "Non-positive dimensions!");
        return -1;
    }
    matrix* pointer_mat = malloc(sizeof(matrix));
    if (!pointer_mat) {
        return -1;
    }
    *mat = pointer_mat;
    double* pointer_data = calloc(rows*cols, sizeof(double));
    if (!pointer_data) {
        return -1;
    }
    (pointer_mat)->data = pointer_data;
    (pointer_mat)->rows = rows;
    (pointer_mat)->cols = cols;
    (pointer_mat)->ref_cnt = 1;
    (pointer_mat)->parent = NULL;
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    
    if (rows < 1 || cols < 1) {
        return -1;
    }

    matrix* pointer_mat = malloc(sizeof(matrix));
    if (!pointer_mat) {
        return -1;
    }
    *mat = pointer_mat;
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->data = from->data + offset;
    (*mat)->parent = from;
    from->ref_cnt += 1;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    

    if (mat) {
        matrix* par = mat->parent;
        int count = mat->ref_cnt;
        if (!par) {
            if (count == 1) {
                free(mat->data);
                free(mat);
            } else {
                mat->ref_cnt = count - 1;
            }
        } else {
            if (par->ref_cnt > 1) {
                free(mat);
                par->ref_cnt -= 1;
            } else {
                free(par->data);
                free(mat->parent);
                free(mat);
            }
        }
    }
    
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    int cols = mat->cols;
    return mat->data[row*cols+col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    int cols = mat->cols;
    mat->data[row * cols + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
   
    int num = (mat->cols) * (mat->rows);
    double* mat_data = mat->data;
    __m256d vector = _mm256_set1_pd(val);
    int num_bottom = num/128 * 128;
    // unsigned int num_bottom2 = (num - num_bottom1)/16 * 16 + num_bottom1;
    // unsigned int num_bottom3 = (num - num_bottom1 - num_bottom2)/4 * 4 + num_bottom1 + num_bottom2;
    #pragma omp parallel for
    for (int i = 0; i < num_bottom; i += 128) {
        _mm256_storeu_pd(mat_data + i, vector);
        _mm256_storeu_pd(mat_data + i+4, vector);
        _mm256_storeu_pd(mat_data + i+8, vector);
        _mm256_storeu_pd(mat_data + i+12, vector);
        _mm256_storeu_pd(mat_data + i+16, vector);
        _mm256_storeu_pd(mat_data + i+20, vector);
        _mm256_storeu_pd(mat_data + i+24, vector);
        _mm256_storeu_pd(mat_data + i+28, vector);
        _mm256_storeu_pd(mat_data + i+32, vector);
        _mm256_storeu_pd(mat_data + i+36, vector);
        _mm256_storeu_pd(mat_data + i+40, vector);
        _mm256_storeu_pd(mat_data + i+44, vector);
        _mm256_storeu_pd(mat_data + i+48, vector);
        _mm256_storeu_pd(mat_data + i+52, vector);
        _mm256_storeu_pd(mat_data + i+56, vector);
        _mm256_storeu_pd(mat_data + i+60, vector);
        _mm256_storeu_pd(mat_data + i+64, vector);
        _mm256_storeu_pd(mat_data + i+68, vector);
        _mm256_storeu_pd(mat_data + i+72, vector);
        _mm256_storeu_pd(mat_data + i+76, vector);
        _mm256_storeu_pd(mat_data + i+80, vector);
        _mm256_storeu_pd(mat_data + i+84, vector);
        _mm256_storeu_pd(mat_data + i+88, vector);
        _mm256_storeu_pd(mat_data + i+92, vector);
        _mm256_storeu_pd(mat_data + i+96, vector);
        _mm256_storeu_pd(mat_data + i+100, vector);
        _mm256_storeu_pd(mat_data + i+104, vector);
        _mm256_storeu_pd(mat_data + i+108, vector);
        _mm256_storeu_pd(mat_data + i+112, vector);
        _mm256_storeu_pd(mat_data + i+116, vector);
        _mm256_storeu_pd(mat_data + i+120, vector);
        _mm256_storeu_pd(mat_data + i+124, vector);
        
    }

    // for (int i = 0; i < num/16 * 16; i += 16) {
    //     _mm256_storeu_pd(mat_data + i, vector);
    //     _mm256_storeu_pd(mat_data + i+4, vector);
    //     _mm256_storeu_pd(mat_data + i+8, vector);
    //     _mm256_storeu_pd(mat_data + i+12, vector);
    // }


    // for (int i = num_bottom2; i < num_bottom3; i += 4) {
    //     _mm256_storeu_pd(mat_data + i, vector);
    // }

    for (int j = num_bottom; j < num; j++) {
            mat_data[j] = val;
        }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    
    int row_1 = result->rows;
    int col_1 = result->cols;
    int row_2 = mat1->rows;
    int col_2 = mat1->cols;
    int row_3 = mat2->rows;
    int col_3 = mat2->cols;
    double* result_data = result->data;
    double* mat1_data = mat1->data;
    double* mat2_data = mat2->data;
    if (row_1 != row_2 || row_2 != row_3 || col_1 != col_2 || col_2 != col_3) {
        return -1;
    }

    int num = row_1 * col_1;
    int num_bottom = num/128 * 128;
    
    #pragma omp parallel for
    for (int i = 0; i < num_bottom; i += 128) {
        int k = i;
       __m256d vector1 = _mm256_loadu_pd(mat1_data + k);
        __m256d vector2 = _mm256_loadu_pd(mat2_data + k);
        __m256d vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
    }

    // unsigned int num_bottom2 = (num - num_bottom1)/16 * 16 + num_bottom1;

    // for (int i = 0; i < num/16 * 16; i += 16) {
    //     int k = i;
    //     __m256d vector1 = _mm256_loadu_pd(mat1_data + k);
    //     __m256d vector2 = _mm256_loadu_pd(mat2_data + k);
    //     __m256d vector3 = _mm256_add_pd(vector1, vector2);
    //     _mm256_storeu_pd(result_data + k, vector3);
    //     k += 4;
    //     vector1 = _mm256_loadu_pd(mat1_data + k);
    //      vector2 = _mm256_loadu_pd(mat2_data + k);
    //      vector3 = _mm256_add_pd(vector1, vector2);
    //     _mm256_storeu_pd(result_data + k, vector3);
    //     k += 4;
    //      vector1 = _mm256_loadu_pd(mat1_data + k);
    //      vector2 = _mm256_loadu_pd(mat2_data + k);
    //      vector3 = _mm256_add_pd(vector1, vector2);
    //     _mm256_storeu_pd(result_data + k, vector3);
    //     k += 4;
    //     vector1 = _mm256_loadu_pd(mat1_data + k);
    //      vector2 = _mm256_loadu_pd(mat2_data + k);
    //      vector3 = _mm256_add_pd(vector1, vector2);
    //     _mm256_storeu_pd(result_data + k, vector3);
    // }

    // unsigned int num_bottom3 = (num - num_bottom1 - num_bottom2)/4 * 4 + num_bottom1 + num_bottom2;

    // for (int i = num_bottom2; i < num_bottom3; i += 4) {
    //     __m256d vector1 = _mm256_loadu_pd(mat1_data + i);
    //     __m256d vector2 = _mm256_loadu_pd(mat2_data + i);
    //     __m256d vector3 = _mm256_add_pd(vector1, vector2);
    //     _mm256_storeu_pd(result_data + i, vector3);
    // }
    
    for (int j = num_bottom; j < num; j++) {
        result_data[j] = mat1_data[j] + mat2_data[j];
    }

    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    int row_1 = result->rows;
    int col_1 = result->cols;
    int row_2 = mat1->rows;
    int col_2 = mat1->cols;
    int row_3 = mat2->rows;
    int col_3 = mat2->cols;
    double* result_data = result->data;
    double* mat1_data = mat1->data;
    double* mat2_data = mat2->data;
    if (row_1 != row_2 || row_2 != row_3 || col_1 != col_2 || col_2 != col_3) {
        return -1;
    }
    
    int num = row_1 * col_1;
    int num_bottom = num/128 * 128;
    // unsigned int num_bottom2 = (num - num_bottom1)/16 * 16 + num_bottom1;
    // unsigned int num_bottom3 = (num - num_bottom1 - num_bottom2)/4 * 4 + num_bottom1 + num_bottom2;
    #pragma omp parallel for
    for (int i = 0; i < num_bottom; i += 128) {
        int k = i;
       __m256d vector1 = _mm256_loadu_pd(mat1_data + k);
        __m256d vector2 = _mm256_loadu_pd(mat2_data + k);
        __m256d vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
        vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
    } 

    // for (int i = 0; i < num/16 * 16; i += 16) {
    //     int k = i;
    //     __m256d vector1 = _mm256_loadu_pd(mat1_data + k);
    //     __m256d vector2 = _mm256_loadu_pd(mat2_data + k);
    //     __m256d vector3 = _mm256_sub_pd(vector1, vector2);
    //     _mm256_storeu_pd(result_data + k, vector3);
    //     k += 4;
    //     vector1 = _mm256_loadu_pd(mat1_data + k);
    //      vector2 = _mm256_loadu_pd(mat2_data + k);
    //      vector3 = _mm256_sub_pd(vector1, vector2);
    //     _mm256_storeu_pd(result_data + k, vector3);
    //     k += 4;
    //      vector1 = _mm256_loadu_pd(mat1_data + k);
    //      vector2 = _mm256_loadu_pd(mat2_data + k);
    //      vector3 = _mm256_sub_pd(vector1, vector2);
    //     _mm256_storeu_pd(result_data + k, vector3);
    //     k += 4;
    //     vector1 = _mm256_loadu_pd(mat1_data + k);
    //      vector2 = _mm256_loadu_pd(mat2_data + k);
    //      vector3 = _mm256_sub_pd(vector1, vector2);
    //     _mm256_storeu_pd(result_data + k, vector3);
    // }

    // for (int i = 0; i < num_bottom3; i += 4) {
    //     __m256d vector1 = _mm256_loadu_pd(mat1_data + i);
    //     __m256d vector2 = _mm256_loadu_pd(mat2_data + i);
    //     __m256d vector3 = _mm256_sub_pd(vector1, vector2);
    //     _mm256_storeu_pd(result_data + i, vector3);
    // }

    for (int j = num_bottom; j < num; j++) {
        result_data[j] = mat1_data[j] - mat2_data[j];
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    
    int row_1 = result->rows;
  
    int col_1 = result->cols;
   
    int row_2 = mat1->rows;
  
    int col_2 = mat1->cols;
  
    int row_3 = mat2->rows;
   
    int col_3 = mat2->cols;


    if (row_1 != row_2 || col_2 != row_3 || col_3 != col_1) {
        return -1;
    }

    double* result_data = result->data;
    double* mat1_data = mat1->data;
    double* mat2_data = mat2->data;
    fill_matrix(result, 0.0);
    
    #pragma omp parallel for
    for (int k = 0; k < row_1; k++) {
        for (int i = 0; i < col_2; i++) {
            for (int j = 0; j < col_1; j++) {
                result_data[k*col_1 + j] += mat1_data[k*col_2 + i] * mat2_data[i*col_3 + j];
            }
        }
    }

    // double global_sum = 0.0;
    // #pragma omp parallel reduction(+: global_sum)
    //  {
    // #pragma omp for
    // for (int i = 0; i < arr_size; i++)
    //   global_sum += x[i] * y[i];
    // }

    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    
    if (mat->rows != mat->cols || mat->rows != result->rows || result->rows != result->cols) {
        return -1;
    }

    if (pow < 0) {
        return -1;
    }

    double* result_data = result->data;
    double* mat_data = mat->data;

    if (pow == 0) {
        int rows = mat->rows;
        #pragma omp parallel for
        for (int i = 0; i < rows*rows; i += (rows+1)) {
            result_data[i] = 1;
        }
        return 0;
    }

    int rows = mat->rows;
    int limit = rows * rows / 4 * 4;

    #pragma omp parallel for
    for (int i = 0; i < limit; i += 4) {
        __m256d vector = _mm256_loadu_pd(mat_data + i);
        _mm256_storeu_pd(result_data + i, vector);
    }

    for (int j = limit; j < rows * rows; j++ ){
        result_data[j] = mat_data[j];
    }

    if (pow == 1) {
        return 0;
    }

    int k = 2;

    
    for (int i = 2; i <= pow; i *= 2) {
        double* data = calloc(rows*rows, sizeof(double));
        matrix tempm = {rows, rows, data, 1, NULL};
        matrix* temp = &tempm;
        add_matrix(temp, temp, result);
       
        if (mul_matrix(result, temp, temp)) {
            free(data);
            return -1;
        }
        free(data);
        k = i;
    }

    
    for (int j = k + 1; j <= pow; j++) {
       
        double* data = calloc(rows*rows, sizeof(double));
        matrix tempm = {rows, rows, data, 1, NULL};
        matrix* temp = &tempm;
        add_matrix(temp, temp, result);
        if (mul_matrix(result, temp, mat)) {
            free(data);
            return -1;
        }
        free(data);
    }
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */

    int rows = mat->rows;
    int cols = mat->cols;
    int rows_r = result->rows;
    int cols_r = result->cols;
    if (rows != rows_r || cols != cols_r) {
        return -1;
    }
    int num = rows*cols;
    double* mat_data = mat->data;
    double* result_data = result->data;
    int num_bottom = num/16 * 16;
    #pragma omp parallel for
    for (int i = 0; i < num_bottom; i += 16) {
        int k = i;
        __m256d vector = _mm256_loadu_pd(mat_data+k);
        __m256d reference = _mm256_set1_pd(0);
         vector = _mm256_sub_pd(reference, vector);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector = _mm256_sub_pd(reference, vector);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector = _mm256_sub_pd(reference, vector);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector = _mm256_sub_pd(reference, vector);
        _mm256_storeu_pd(result_data+k, vector);
    }
    for (int j = num_bottom; j < num; j++) {
        result_data[j] = -mat_data[j];
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
 
    int rows = mat->rows;
    int cols = mat->cols;
    int rows_r = result->rows;
    int cols_r = result->cols;
    if (rows != rows_r || cols != cols_r) {
        return -1;
    }
    int num = rows*cols;
    double* mat_data = mat->data;
    double* result_data = result->data;
    int num_bottom = num/16 * 16;

    #pragma omp parallel for
    for (int i = 0; i < num_bottom; i += 16) {
        int k = i;
        __m256d vector = _mm256_loadu_pd(mat_data+k);
        __m256d reference = _mm256_set1_pd(0);
        __m256d vector_1 = _mm256_sub_pd(reference, vector);
         vector = _mm256_max_pd (vector, vector_1);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector_1 = _mm256_sub_pd(reference, vector);
         vector = _mm256_max_pd (vector, vector_1);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector_1 = _mm256_sub_pd(reference, vector);
         vector = _mm256_max_pd (vector, vector_1);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector_1 = _mm256_sub_pd(reference, vector);
         vector = _mm256_max_pd (vector, vector_1);
        _mm256_storeu_pd(result_data+k, vector);
    }
    for (int j = num_bottom; j < num; j++) {
        result_data[j] = fmax(mat_data[j], -mat_data[j]);
    }
    return 0;
}

