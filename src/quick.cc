#include"../include/quick.h"

int Partion(vector<double> *V, int low, int high)
{
    float piv = (*V)[low];
    while (low < high)
    {
        while (low < high && (*V)[high] >= piv)
        {
            --high;
        }
        (*V)[low] = (*V)[high];
        while (low < high && (*V)[low] <= piv)
        {
            ++low;
        }
        (*V)[high] = (*V)[low];
    }
    (*V)[low]=piv;
    return low;
}
int Partion(double *V, int low, int high)
{
    float piv =*(V+low);
    while (low < high)
    {
        while (low < high && *(V+high) >= piv)
        {
            --high;
        }
        *(V+low)= *(V+high);
        while (low < high && *(V+low) <= piv)
        {
            ++low;
        }
        *(V+high) = *(V+low);
    }
    *(V+low)=piv;
    return low;
}
/*
void QuickSort(vector<double> *V, int low, int high)
{
    
    if (low < high)
    {
        int pivloc = Partion(V, low, high);
        QuickSort(V, low, pivloc - 1);
        QuickSort(V, pivloc + 1, high);
    }
}
*/
void QuickSort(double *V, int low, int high)
{
    
    if (low < high)
    {
        int pivloc = Partion(V, low, high);
        QuickSort(V, low, pivloc - 1);
        QuickSort(V, pivloc + 1, high);
    }
}
