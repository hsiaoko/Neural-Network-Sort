#include "map.h"

void ToBucket(Eigen::MatrixXf input, InitResut *initResult)
{
    for (int i = 0; i < input.rows(); i++)
    {
        int pos = int(round(input(i, 1)));
        if ((*initResult->sortedList)[pos] == FLT_MAX)
        {
            (*initResult->sortedList)[pos*2] = input(i, 0);
        }
        else
        {
            initResult->waitedList->push_back(input(i, 0));
        }
    }
    // cout<<"-------------------------sorted--------------------\n";
    // for (int i = 0; i < initResult->sortedList->size(); i++)
    // {
    //     cout << (*initResult->sortedList)[i] << endl;
    // }
    // cout<<"-------------------------waited--------------------\n";
    // for (int j = 0; j < initResult->waitedList->size(); j++)
    // {
    //     cout << (*initResult->waitedList)[j] << endl;
    // }
    cout << "size of waited:" << initResult->waitedList->size() << endl;
    cout << "size of sorted:" << initResult->sortedList->size() << endl;
}
void Merge(InitResut *initResult, vector<float> *finalResult)
{
    int indexa = 0;
    int indexb = 0;
    while (initResult->sortedList->size() > indexa && initResult->waitedList->size() > indexb)
    {
        if ((*initResult->sortedList)[indexa]!=FLT_MAX)
        {
            if ((*initResult->sortedList)[indexa] <= (*initResult->waitedList)[indexb])
            {
                finalResult->push_back((*initResult->sortedList)[indexa]);
                indexa += 1;
            }
            else
            {
                finalResult->push_back((*initResult->waitedList)[indexb]);
                indexb += 1;
            }
        }
        else
        {
            ++indexa;
        }
        
    }
    finalResult->insert(finalResult->end(), initResult->sortedList->begin() + indexa, initResult->sortedList->end());
    finalResult->insert(finalResult->end(), initResult->waitedList->begin() + indexb, initResult->waitedList->end());
    cout << "final:" << finalResult->size() << endl;
}
