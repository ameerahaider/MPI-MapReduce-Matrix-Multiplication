#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void create_matrices(int matrix_size, const char *fileA, const char *fileB);
void load_matrix(const char *filename, int *matrix, int size);
void save_matrix(const char *filename, int *matrix, int size);

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Process ID %d running on %s\n", rank, processor_name);

    int matrix_size = atoi(argv[1]);
    int num_mappers = (size - 1) * 2 / 3;
    int num_reducers = size - 1 - num_mappers;

    const char *input_file_A = argv[2];
    const char *input_file_B = argv[3];
    const char *output_file = argv[4];

    int *A = (int *)malloc(matrix_size * matrix_size * sizeof(int));
    int *B = (int *)malloc(matrix_size * matrix_size * sizeof(int));
    int *C = (int *)calloc(matrix_size * matrix_size, sizeof(int));

    create_matrices(matrix_size, input_file_A, input_file_B);

    load_matrix(input_file_A, A, matrix_size);
    load_matrix(input_file_B, B, matrix_size);

    typedef struct
    {
        int row;
        int col;
        int value;
    } KeyValue;

    if (rank == 0)
    {

        printf("Inside Master with process ID %d running on %s\n", rank, processor_name);

        if (argc < 5)
        {
            MPI_Finalize();
            return 1;
        }

        if (matrix_size < 2)
        {
            if (rank == 0)
            {
                printf("Matrix size must be at least 2\n");
            }
            MPI_Finalize();
            return 1;
        }

        for (int i = 1; i < size; i++)
        {
            int task = (i <= num_mappers) ? 1 : 2;
            MPI_Send(&task, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            printf("Task %s assigned to process %d\n", (task == 1) ? "Map" : "Reduce", i);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int rows_per_mapper = matrix_size / num_mappers;
        int cols_per_reducer = matrix_size / num_reducers;

        for (int i = 1; i <= num_mappers; i++)
        {
            int start_row = (i - 1) * rows_per_mapper;
            int end_row = start_row + rows_per_mapper;

            MPI_Send(&start_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&end_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

            printf("Mapper %d assigned rows %d to %d\n", i, start_row, end_row);
        }

        KeyValue *intermediate_data = (KeyValue *)malloc(matrix_size * matrix_size * sizeof(KeyValue));
        int intermediate_data_count = 0;

        for (int i = 1; i <= num_mappers; i++)
        {
            for (int j = 0; j < matrix_size * rows_per_mapper; j++)
            {
                KeyValue kv;
                MPI_Recv(&kv.row, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&kv.col, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&kv.value, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                intermediate_data[intermediate_data_count++] = kv;
            }
        }

        for (int i = 0; i < intermediate_data_count; i++)
        {
            int target_reducer = num_mappers + ((intermediate_data[i].col / cols_per_reducer) % num_reducers) + 1;
            MPI_Send(&intermediate_data[i].row, 1, MPI_INT, target_reducer, 0, MPI_COMM_WORLD);
            MPI_Send(&intermediate_data[i].col, 1, MPI_INT, target_reducer, 0, MPI_COMM_WORLD);
            MPI_Send(&intermediate_data[i].value, 1, MPI_INT, target_reducer, 0, MPI_COMM_WORLD);
        }

        free(intermediate_data);

        for (int i = num_mappers + 1; i <= num_mappers + num_reducers; i++)
        {
            int start_col = (i - num_mappers - 1) * cols_per_reducer;
            int end_col = start_col + cols_per_reducer;
            MPI_Send(&start_col, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&end_col, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        for (int i = num_mappers + 1; i <= num_mappers + num_reducers; i++)
        {
            for (int j = 0; j < matrix_size; j++)
            {
                for (int k = 0; k < cols_per_reducer; k++)
                {
                    int row, col, value;
                    MPI_Recv(&row, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&col, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&value, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    C[row * matrix_size + col] = value;
                }
            }
        }

        save_matrix(output_file, C, matrix_size);
        printf("Matrix multiplication complete.\n");
    }
    else
    {

        MPI_Barrier(MPI_COMM_WORLD);
        printf("Inside Process %d running on %s\n", rank, processor_name);

        int task;
        int value;
        MPI_Recv(&task, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        char hostname[MPI_MAX_PROCESSOR_NAME];
        int len;
        MPI_Get_processor_name(hostname, &len);

        printf("Process %d received task %s on %s\n", rank, (task == 1) ? "Map" : "Reduce", hostname);

        int rows_per_mapper = matrix_size / num_mappers;
        int cols_per_reducer = matrix_size / num_reducers;

        if (task == 1)
        {

            printf("In the Mapper for process %d\n", rank);

            int start_row, end_row;
            MPI_Recv(&start_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&end_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = start_row; i < end_row; i++)
            {
                for (int j = 0; j < matrix_size; j++)
                {
                    int sum = 0;
                    for (int k = 0; k < matrix_size; k++)
                    {
                        sum += A[i * matrix_size + k] * B[k * matrix_size + j];
                    }
                    MPI_Send(&i, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(&j, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(&sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                }
            }
        }
        else if (task == 2)
        {

            printf("In the Reducer for process %d\n", rank);

            KeyValue *intermediate_data = (KeyValue *)malloc(matrix_size * cols_per_reducer * sizeof(KeyValue));
            int intermediate_data_count = 0;

            for (int i = 0; i < matrix_size * cols_per_reducer; i++)
            {
                KeyValue kv;
                MPI_Recv(&kv.row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&kv.col, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&kv.value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                intermediate_data[intermediate_data_count++] = kv;
            }

            int start_col, end_col;
            MPI_Recv(&start_col, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&end_col, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < matrix_size; i++)
            {
                for (int j = start_col; j < end_col; j++)
                {
                    int sum = 0;
                    for (int k = 0; k < intermediate_data_count; k++)
                    {
                        if (intermediate_data[k].col == j && intermediate_data[k].row == i)
                        {
                            sum += intermediate_data[k].value;
                        }
                    }
                    MPI_Send(&i, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(&j, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(&sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                }
            }

            free(intermediate_data);
        }
    }

    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}

void create_matrices(int matrix_size, const char *fileA, const char *fileB)
{
    srand(time(NULL));
    int *A = (int *)malloc(matrix_size * matrix_size * sizeof(int));
    int *B = (int *)malloc(matrix_size * matrix_size * sizeof(int));
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            A[i * matrix_size + j] = rand() % 10;
            B[i * matrix_size + j] = rand() % 10;
        }
    }

    save_matrix(fileA, A, matrix_size);
    save_matrix(fileB, B, matrix_size);

    free(A);
    free(B);
}

void load_matrix(const char *filename, int *matrix, int size)
{
    FILE *file = fopen(filename, "r");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            fscanf(file, "%d", &matrix[i * size + j]);
        }
    }
    fclose(file);
}

void save_matrix(const char *filename, int *matrix, int size)
{
    FILE *file = fopen(filename, "w");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            fprintf(file, "%d ", matrix[i * size + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}