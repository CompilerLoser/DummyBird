#include <iostream>
#include "utils/helper.h"
void foo()
{
    std::cout << "test" << std::endl;
}

bool conflict(int num, int row)
{
    return true;
}

static bool in_res(int num, int *row, int size)
{
    for (int i = 0; i < size; ++i)
        if (row[i] == num)
            return true;
    return false;
}

void generate_random_cols(int range, int len, int rand_size, bool (*fn)(int, int), int *res)
{
    for (int i = 0; i < len; ++i)
    {
        for (int j = 0; j < rand_size; ++j)
        {
            int rand_value = random(range);
            while (in_res(rand_value, &res[i * rand_size], rand_size) || !fn(rand_value, i))
            {
                rand_value = random(range);
            }
            res[i * rand_size + j] = rand_value;
        }
    }
}
void rand_init_matrix(int range, int row, int col, float * M)
{
    for(int i =0; i< row; ++i)
        for(int j =0; j<col; ++j)
            M[i*col+j] = float(random(range));
}