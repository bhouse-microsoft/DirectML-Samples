#include "pch.h"

#include "FloatMath.h"
#include <stdlib.h>
#include <math.h>

double randd()
{
    return abs(static_cast <double> (rand()) / static_cast <double> (RAND_MAX));
}

template<int precision>
ufloat<precision> Sum(std::vector<ufloat<precision>> & a)
{
    ufloat<precision> sum = ufloat<precision>(0.0);
    for (size_t i = 0; i < a.size(); i++) {
        double a_value = a[i].Get();
        double s_value = sum.Get();
        s_value += a_value;
        sum = ufloat<precision>(s_value);
    }

    return sum;
}

double Sum(std::vector<double> & a)
{
    double sum = 0;
    for (size_t i = 0; i < a.size(); i++)
        sum += a[i];

    return sum;
}

template<int precision>
ufloat<precision> DotProduct(std::vector<ufloat<precision>> & a, std::vector<ufloat<precision>> & b)
{
    std::vector<double> products;

    assert(a.size() == b.size());

    products.resize(a.size());

    for (size_t i = 0; i < a.size(); i++) {
        double a_value = a[i].Get();
        double b_value = b[i].Get();
        double product = a_value * b_value;
        products[i] = product;
    }

    ufloat<precision> sum = ufloat<precision>(0.0);
    for (size_t i = 0; i < a.size(); i++) {
        double product = products[i];
        ufloat<precision> p = ufloat<precision>(product);
        double p_value = p.Get();
        double s_value = sum.Get();
        s_value += p_value;
        sum = ufloat<precision>(s_value);
    }

    return sum;
}

double DotProduct(std::vector<double> & a, std::vector<double> & b)
{
    double sum = 0;

    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); i++)
        sum += a[i] * b[i];

    return sum;
}


void TestDotProduct()
{
    for (int i = 0; i < 100000; i++)
    {
        const static int low_precision = 23;
        const static int double_precision = 52;
        const static int length = 1000;
        const static int bitsUsedForAccumulation = 10; // (log base 2 of 1024)

        double ulp = 1.0f / (1 << (low_precision - 1));
        double variance = 16.0 * ulp;

        std::vector<ufloat<low_precision>> a_low;
        std::vector<ufloat<low_precision>> b_low;

        std::vector<ufloat<double_precision>> a_double;
        std::vector<ufloat<double_precision>> b_double;

        std::vector<double> a;
        std::vector<double> b;

        a_low.resize(length);
        b_low.resize(length);

        a_double.resize(length);
        b_double.resize(length);

        a.resize(length);
        b.resize(length);

        for (int i = 0; i < length; i++) {
            double a_value = randd();
            double delta = variance * (1.0 - (2.0 * randd()));
            double b_value = (1.0 + delta) / a_value;

            a[i] = a_value;
            b[i] = b_value;

            a_low[i] = ufloat<low_precision>(a_value);
            b_low[i] = ufloat<low_precision>(b_value);

            a_double[i] = ufloat<double_precision>(a_value);
            b_double[i] = ufloat<double_precision>(b_value);
        }

        ufloat<low_precision> r_no_sort = DotProduct<low_precision>(a_low, b_low);

        double r = DotProduct(a, b);

        double no_sort_error = abs(r_no_sort.Get() - r);

        if (no_sort_error > ((double)length * ulp * 2.0)) {
            printf("reference %.10f\n", r);
            printf("results = %.10f\n", r_no_sort.Get());

            printf("no sort:%.10f/%.3f\n", no_sort_error, no_sort_error / ((double)length * ulp));
        }

    }

    printf("done\n");

}

void TestSum()
{
    const static int low_precision = 23;
    const static int lower_precision = 22;
    const static int double_precision = 52;
    const static int length = 1024;
    const static int bitsUsedForAccumulation = 11; // (log base 2 of 2048)
    double estimated_sum = 2048.0f;
    double estimated_exponent = log2(estimated_sum);
    double ulp = pow(2.0f, estimated_exponent - low_precision + 1);
    double estimated_max_error = ulp * sqrt(length) / 2;
    double max_error = 0.0f;
    double error_sum = 0.0f;
    int trials = 10000;

    for (int i = 0; i < trials; i++)
    {
        std::vector<ufloat<lower_precision>> a_low;
        std::vector<ufloat<double_precision>> a_double;

        std::vector<double> a;

        a_low.resize(length);
        a_double.resize(length);
        a.resize(length);

        for (int i = 0; i < length; i++) {
            double a_value = 1.0 + randd();
            a[i] = a_value;
            a_low[i] = ufloat<lower_precision>(a_value);
            a_double[i] = ufloat<double_precision>(a_value);
        }

        ufloat<lower_precision> sum_low = Sum<lower_precision>(a_low);

        double sum = Sum(a);

        double low_error = abs(sum_low.Get() - sum);

        if (low_error > max_error) max_error = low_error;

        //        printf("reference %.10f\n", sum);
        //        printf("result = %.10f\n", sum_low.Get());
        //        printf("error = %.10f\n", low_error);

        error_sum += low_error;
    }

    printf("max error = %.10f\n", max_error);
    printf("avg error = %.10f\n", error_sum / (double)trials);
    printf("estimated max error = %.10f\n", estimated_max_error);

}
