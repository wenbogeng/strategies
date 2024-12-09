#include <map>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "Eigen/Dense"


using namespace std;


/*
 
 ################################################################################################
 '''Data Structures'''
 ################################################################################################
 
 */

//structure for a cell in DataFrame
struct StockData {
    bool double_or_string;  //false=double,  true=string
    string str;
    double value;
    
    StockData(double _value) {
        double_or_string = false;
        value = _value;
    }
    
    StockData(string _value) {
        double_or_string = true;
        str = _value;
        value = 0.0;
    }
    StockData(const StockData& o) {
        double_or_string = o.double_or_string;
        str = o.str;
        value = o.value;
    }
    StockData& operator=(const StockData& o) {
        this->double_or_string = o.double_or_string;
        this->str = o.str;
        this->value = o.value;
        return *this;
    }
    ~StockData() {}
};

//structure for a row in DataFrame
//the i-th item corresponds to the i-th column of the DataFrame
typedef vector<StockData> DataRow;

//structure for the DataFrame
struct DataFrame {
    vector<string> column_names; //the names of columns; the i-th name corresponds to the i-th column of the DataFrame
    vector<DataRow> data; //the rows in the DataFrame
    map<string, bool> column_type;
    
    DataFrame() {}
    
    DataFrame(const DataFrame& o, bool copydata = true) {
        column_names = o.column_names;
        column_type = o.column_type;
        
        if(copydata) {
            data = o.data;
        }
    }
};

// Helper structure to keep track of values and their original indices
struct ValIndex {
    double value;
    int originalIndex;
    ValIndex(double v, int i) : value(v), originalIndex(i) {}
};



/*
 
 ################################################################################################
 '''Helper Functions'''
 ################################################################################################
 
 */

//if two double equal
bool compfloat(double a, double b) {
    double E = 0.0000001;
    return a > b - E && a < b + E;
}


//if a < b
bool floatLess(double a, double b) {
    double E = 0.0000001;
    return a < b - E;
}


// Helper function to split a string by a delimiter and return a vector of tokens
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}


// read DataFrame from csv file
DataFrame readCSV(const char* path) {
    std::ifstream file(path);
    std::string line;
    
    // Check if file is open
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        throw std::runtime_error("file not found");
    }
    
    // Read the header line first if you want to skip it or process header names
    if (std::getline(file, line)) {}
    
    
    DataFrame df;
    df.column_names = vector<string>{"stock_id", "date", "return", "factor", "market_value", "industry"};
    vector<bool> column_type{true, true, false, false, false, true};
    for(size_t i = 0; i < column_type.size(); ++i) {
        df.column_type[df.column_names[i]] = column_type[i];
    }
    
    // Read data lines
    while (std::getline(file, line)) {
        std::vector<std::string> tokens = split(line, ',');
        // Now you have the tokens for one line, you can process them
        
        DataRow row;
        for (size_t i = 0; i < tokens.size(); ++i) {
            //std::cout << tokens[i];
            //if (i < tokens.size() - 1) std::cout << ", ";
            size_t l = tokens[i].length();
            if(tokens[i][l - 1] == '\r' || tokens[i][l - 1] == '\n') {
                tokens[i] = tokens[i].substr(0, l - 1);
            }
            bool ctype = column_type[i];
            if (!ctype) {
                std::istringstream iss(tokens[i]);
                double value;
                iss >> value;
                StockData item(value);
                row.push_back(item);
            }
            else {
                row.push_back(tokens[i]);
            }
        }
        df.data.push_back(row);
    }
    
    file.close();
    return df;
}



//return the type(false=double;true=string) of a column using column name
bool getColumnType(DataFrame& df, string colname) {
    if (df.column_type.find(colname) == df.column_type.end()) {
        return false;
    }
    return df.column_type[colname];
    
}


//return the position of a column using column name
int getColumnIdx(const DataFrame& df, const string& column_name) {
    for (size_t i = 0; i < df.column_names.size(); ++i) {
        if (df.column_names[i] == column_name)
            return (int)i;
    }
    return -1; // Not found
}



//create a new row in DataFrame
DataRow createRow(DataFrame& result) {
    DataRow newRow;
    for(size_t i = 0; i < result.column_names.size(); ++i) {
        string colname = result.column_names[i];
        bool type = result.column_type[colname];
        if(type) {
            
            StockData item("");
            newRow.push_back(item);
        }
        else {
            StockData item(0.0);
            newRow.push_back(item);
        }
    }
    return newRow;
    
}



//add a column with column name and column data  to a DataFrame;  this column will be the last in column list
DataFrame addColumn(DataFrame& df, string columnName, const vector<StockData>& columnData, bool coltype) {
    // First, add the new column name
    df.column_names.push_back(columnName);
    
    // Check if we need to initialize rows (in case DataFrame was empty)
    if (df.data.empty()) {
        df.data.resize(columnData.size());
    }
    
    // Then, add the data to each row
    for (size_t i = 0; i < df.data.size() && i < columnData.size(); ++i) {
        df.data[i].push_back(columnData[i]);
    }
    
    df.column_type[columnName] = coltype;
    return df;
}


//get all data of a column in DataFrame
vector<StockData> selectColumn(const DataFrame& df, const string& column_name) {
    vector<StockData> column;
    int idx = getColumnIdx(df, column_name);
    if (idx != -1) {
        for (size_t i = 0; i < df.data.size(); ++i) {
            column.push_back(df.data[i][idx]);
        }
    }
    return column;
}



// get all data of multiple columns in a DataFrame
DataFrame selectColumns(DataFrame& df, const vector<string>& column_names) {
    DataFrame res;
    
    for(size_t i = 0; i < column_names.size(); ++i) {
        string name = column_names[i];
        vector<StockData> column_data = selectColumn(df, name);
        addColumn(res, name, column_data, df.column_type[name]);
    }
    
    return res;
}


//change the name of a column in DataFrame
void renameColumn(DataFrame& df, string oldname, string newname) {
    int idx = getColumnIdx(df, oldname);
    df.column_names[idx] = newname;
    df.column_type[newname] = df.column_type[oldname];
    df.column_type.erase(oldname);
}


// Function to compare StockData
bool compareStockData(const StockData& a, const StockData& b) {
    if (a.double_or_string && b.double_or_string) { // Both strings
        return a.str < b.str;
    } else if (!a.double_or_string && !b.double_or_string) { // Both doubles
        return floatLess(a.value, b.value);
    }
    return false; // One is double, the other is string
}


//split a DataFrame into groups
vector<DataFrame> groupBy(DataFrame& df, string column_name) {
    int idx = getColumnIdx(df, column_name);
    if (idx == -1) {
        throw std::runtime_error("Column name not found");
    }
    
    // Map to hold grouped rows keyed by StockData
    map<StockData, vector<DataRow>, bool(*)(const StockData&, const StockData&)> groupedRows(compareStockData);
    map<StockData, vector<DataRow>, bool(*)(const StockData&, const StockData&)>::iterator it;
    // Populate the map
    for (size_t i = 0; i < df.data.size(); ++i) {
        DataRow& row = df.data[i];
        StockData key = row[idx];
        groupedRows[key].push_back(row);
    }
    
    // Convert the map into a vector of DataFrames
    vector<DataFrame> groupedDataFrames;
    for (it = groupedRows.begin(); it != groupedRows.end(); ++it) {
        
        DataFrame groupDf(df, false);
        groupDf.data = it->second;
        groupedDataFrames.push_back(groupDf);
    }
    
    return groupedDataFrames;
}


// Helper function to calculate the median of a vector
double median(vector<double>& v) {
    size_t n = v.size();
    if (n == 0) return 0.0; // Handle empty vector case
    
    sort(v.begin(), v.end());
    if (n % 2 == 0) {
        return (v[n / 2 - 1] + v[n / 2]) / 2.0;
    } else {
        return v[n / 2];
    }
}


// Helper function to calculate the mean of a vector
double mean(const vector<double>& v) {
    if (v.empty()) return 0.0;
    return accumulate(v.begin(), v.end(), 0.0) / v.size();
}


// Helper function to calculate the standard deviation of a vector
double stddev(const vector<double>& v, double m) {
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += (v[i] - m) * (v[i] - m);
    }
    return sqrt(sum / (v.size() - 1));
}


//display the content of a DataFrame
void print(DataFrame& df, int count, const char* name)
{
    if (count < 0) count = 30;
    size_t start = 0;//df.data.size() - count;
    if (df.data.size() < count) {
        start = 0;
    }
    
    printf("-------------------------- DataFrame %s -----------------------\n", name);
    size_t colcount = df.column_names.size();
    
    for(size_t i = 0; i < colcount; ++i) {
        printf("%s\t", df.column_names[i].c_str());
    }
    printf("\n");
    
    for(size_t i = 0; i < count && i < df.data.size(); ++i) {
        for(size_t j = 0; j < colcount; ++j) {
            bool coltype = df.column_type[df.column_names[j]];
            
            if(coltype) {
                printf("%s\t", df.data[i + start][j].str.c_str());
            }
            else {
                printf("%lf\t", df.data[i + start][j].value);
            }
        }
        printf("\n");
    }
    printf("\n\n\n");
}


/*
 
 ################################################################################################
 '''Standardization, shrinking extreme value processing'''
 ################################################################################################
 
 */


// Function to apply winsorization to the 'factor' column of a DataFrame
void winsorize(DataFrame& df, int scale = 5) {
    vector<double> factors;
    
    int factorIdx = getColumnIdx(df, "factor");
    // Extract 'factor' values
    for (size_t i = 0; i < df.data.size(); ++i) {
        factors.push_back(df.data[i][factorIdx].value);
    }
    
    // Calculate median and MAD of 'factor' values
    double med = median(factors);
    for (size_t i = 0; i < factors.size(); ++i) {
        factors[i] = std::abs(factors[i] - med);
    }
    double mad = median(factors);
    
    // Calculate bounds for winsorization
    double lower_bound = med - scale * mad;
    double upper_bound = med + scale * mad;
    
    
    int idx1 = getColumnIdx(df, "factor");
    
    // Apply winsorization to the 'factor' field
    for (size_t i = 0; i < df.data.size(); ++i) {
        df.data[i][idx1].value = std::min(std::max(df.data[i][factorIdx].value, lower_bound), upper_bound);
    }
}

// Function to standardize the 'factor' column of a DataFrame
void standardize(DataFrame& df) {
    vector<double> factors;
    
    int idx1 = getColumnIdx(df, "factor");
    
    
    // Extract 'factor' values
    for (size_t i = 0; i < df.data.size(); ++i) {
        factors.push_back(df.data[i][idx1].value);
    }
    
    // Calculate mean and standard deviation of 'factor' values
    double m = mean(factors);
    double s = stddev(factors, m);
    
    // Standardize 'factor' values if standard deviation is not zero
    if (s != 0) {
        for (size_t i = 0; i < df.data.size(); ++i) {
            df.data[i][idx1].value = (df.data[i][idx1].value - m) / s;
        }
    }
    // If standard deviation is zero, 'factor' values remain unchanged as per Python code comment
}


// Function to generate dummy variables for a vector of categorical strings
map<string, vector<int>> get_dummies(const vector<StockData>& categories) {
    map<string, vector<int>> dummies;
    map<string, int> categoryIndex;
    
    // Initialize the categoryIndex map and dummies' vectors with zeros
    for (size_t i = 0; i < categories.size(); ++i) {
        if (categoryIndex.find(categories[i].str) == categoryIndex.end()) {
            categoryIndex[categories[i].str] = (int)i; // Map each category to its index
            vector<int> dummyVec(categories.size(), 0); // Initialize a vector of zeros for each category
            dummies[categories[i].str] = dummyVec;
        }
    }
    
    // Fill the dummy vectors
    for (size_t i = 0; i < categories.size(); ++i) {
        dummies[categories[i].str][i] = 1;
    }
    
    return dummies;
}


Eigen::MatrixXd mapToMatrixXd(std::map<std::string, std::vector<int>>& data) {
    // Determine the size of the matrix
    size_t rows = data.size();
    size_t cols = 0;
    
    std::map<std::string, std::vector<int>>::iterator entry;
    for (entry = data.begin(); entry !=  data.end(); ++entry) {
        
        if (entry->second.size() > cols) {
            cols = entry->second.size();
        }
    }
    
    // Create a temporary matrix to hold the data
    Eigen::MatrixXd tempMatrix(rows, cols);
    
    // Copy data from the map to the temporary matrix
    int row = 0;
    for (entry = data.begin(); entry !=  data.end(); ++entry) {
        for (size_t i = 0; i < entry->second.size(); ++i) {
            tempMatrix(row, i) = entry->second[i];
        }
        ++row;
    }
    
    return tempMatrix;
}

// Helper function to find unique quantile thresholds
vector<double> findUniqueQuantiles(vector<double>& sorted_arr, int group_num) {
    vector<double> thresholds;
    for (int i = 1; i <= group_num; ++i) {
        double quantileValue = sorted_arr[(i * sorted_arr.size()) / group_num - 1];
        thresholds.push_back(quantileValue);
    }
    // Remove duplicates
    sort(thresholds.begin(), thresholds.end());
    vector<double>::iterator it = unique(thresholds.begin(), thresholds.end());
    thresholds.resize(distance(thresholds.begin(), it));
    return thresholds;
}

// Comparison function for sorting
bool compareValIndex(const ValIndex& a, const ValIndex& b) {
    return a.value < b.value;
}


/*
 
 ################################################################################################
 '''Industry neutral and market capitalization neutral treatment'''
 ################################################################################################
 
 */



void neutralize0(DataFrame& df, const Eigen::MatrixXd& industry_dummies) {
    vector<DataRow>& data = df.data;
    int idx1 = getColumnIdx(df, "market_value");
    int idx2 = getColumnIdx(df, "factor");
    
    size_t n = data.size();
    size_t m = industry_dummies.cols(); // The number of industry dummy variables
    
    // Construct X matrix: constant terms, logarithmic market capitalization
    Eigen::MatrixXd X(n, m + 2);
    X.col(0) = Eigen::VectorXd::Ones(n); // Constant term
    for(size_t i = 0; i < n; ++i) {
        X(i, 1) = std::log(data[i][idx1].value); // Logarithmic market capitalization
    }
    X.block(0, 2, n, m) = industry_dummies; // 行业哑变量
    
    // Construct factor value vector
    Eigen::VectorXd y(n);
    for(size_t i = 0; i < n; ++i) {
        y(i) = data[i][idx2].value;
    }
    
    // Perform regression analysis
    Eigen::VectorXd coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    Eigen::VectorXd y_fit = X * coefficients;
    Eigen::VectorXd residuals = y - y_fit; // Residuals are used as neutralized factor values
    
    vector<StockData> arr;
    // Update factor value
    for(size_t i = 0; i < n; ++i) {
        
        arr.push_back(StockData(residuals(i)));
    }
    
    addColumn(df, "factor_n", arr, false);
}


void neutralize(DataFrame& df) {
    vector<StockData> industry = selectColumn(df, "industry");
    
    map<string, vector<int>> dummy = get_dummies(industry);
    
    Eigen::MatrixXd mat = mapToMatrixXd(dummy);
    
    neutralize0(df, mat.transpose());
}


/*
 
 ################################################################################################
 '''Indicator Calculation Functions'''
 ################################################################################################
 
 */

//convert a date string like "2020-01-01" into the number of days passed since epoch
int date2day(string date) {
    struct tm timeStruct;
    memset(&timeStruct, 0, sizeof(timeStruct));
    strptime(date.c_str(), "%Y-%m-%d", &timeStruct);
    
    // Convert tm structure to time_t and then to days
    // Assuming Unix timestamp starts from 1970-01-01, which is considered as day 0
    time_t timeInSeconds = mktime(&timeStruct);
    //printf("%ld\n" , timeInSeconds);
    int days = static_cast<int>(timeInSeconds / (24 * 60 * 60));
    //printf("%d\n" , days);
    return days;
}

//implement the function of 'pandas.qcut()' in python
vector<int> qcutWithDrop(vector<double>& arr, int group_num) {
    vector<int> labels(arr.size(), 0);
    if (arr.empty() || group_num <= 0) return labels;
    
    // Sort a copy of the array to find quantile thresholds without altering original array
    vector<double> sorted_arr = arr;
    sort(sorted_arr.begin(), sorted_arr.end());
    
    // Find unique quantile thresholds
    vector<double> thresholds = findUniqueQuantiles(sorted_arr, group_num);
    
    // Assign labels based on unique thresholds
    for (size_t i = 0; i < arr.size(); ++i) {
        int label = 0;
        for (size_t j = 0; j < thresholds.size(); ++j) {
            if (arr[i] > thresholds[j]) {
                label = (int)(j + 1); // Assign to the next bin if value is greater than the current threshold
            } else {
                break; // Stop checking further thresholds
            }
        }
        labels[i] = label;
    }
    
    return labels;
}


//implement the function of 'pandas.qcut()' in python
vector<int> qcut(vector<double>& arr, int group_num) {
    int n = (int)arr.size();
    vector<ValIndex> sortedArr;
    sortedArr.reserve(n);
    
    // Fill the helper structure
    for (int i = 0; i < n; ++i) {
        sortedArr.push_back(ValIndex(arr[i], i));
    }
    
    // Sort the array based on values
    sort(sortedArr.begin(), sortedArr.end(), compareValIndex);
    
    // Prepare the result vector
    vector<int> result(n, 0);
    int groupSize = n / group_num;
    int remainder = n % group_num; // Extra elements to distribute among the first 'remainder' groups
    
    // Assign group numbers
    for (int i = 0; i < group_num; ++i) {
        int start = i * groupSize + min(i, remainder);
        int end = start + groupSize + (i < remainder ? 1 : 0);
        for (int j = start; j < end; ++j) {
            result[sortedArr[j].originalIndex] = i;
        }
    }
    
    return result;
}


/*
 
 ################################################################################################
 '''Multi-Factor Portfolio Backtesting Functions'''
 ################################################################################################
 
 */


//resample DataFrame based on 'date' column
vector<DataFrame> resample(DataFrame& df, int interval_days) {
    vector<DataFrame> resampled;
    map<int, DataFrame> grouped;
    
    
    int idx = getColumnIdx(df, "date");
    int day0 = INT_MAX;
    for (size_t i = 0; i < df.data.size(); ++i) {
        int day = date2day(df.data[i][idx].str); // Determine the period/bin
        if(day < day0) day0 = day;
    }
    
    for (size_t i = 0; i < df.data.size(); ++i) {
        int day = date2day(df.data[i][idx].str); // Determine the period/bin
        int period = (day - day0) / interval_days;
        //printf("%s %s %ld\n" , df.data[i][0].str.c_str(), df.data[i][1].str.c_str(), period);
        if(grouped.find(period) == grouped.end()) {
            grouped[period] = DataFrame(df, false);
        }
        grouped[period].data.push_back(df.data[i]);
    }
    
    for (map<int, DataFrame>::iterator it = grouped.begin(); it != grouped.end(); ++it) {
        resampled.push_back(it->second);
    }
    
    return resampled;
}



//take the first row of every dataframe in 'dfs'
DataFrame first(vector<DataFrame>& dfs) {
    DataFrame result(dfs[0], false);
    for (size_t i = 0; i < dfs.size(); ++i) {
        if (!dfs[i].data.empty()) {
            result.data.push_back(dfs[i].data[0]); // Take the first StockData of each DataFrame
        }
    }
    return result;
}


//drop a column like what the python code does:  df.drop([column_name],axis=1)
DataFrame dropColumn(DataFrame& df, const string& columnName) {
    DataFrame result; // Create a new DataFrame to hold the result
    
    // Find the index of the column to be dropped
    int dropIndex = -1;
    for (size_t i = 0; i < df.column_names.size(); ++i) {
        if (df.column_names[i] == columnName) {
            dropIndex = (int)i;
            break;
        }
    }
    
    // If column was not found, return the original DataFrame
    if (dropIndex == -1)
        return df;
    
    
    // Copy column names excluding the one to be dropped
    for (size_t i = 0; i < df.column_names.size(); ++i) {
        if (i != static_cast<size_t>(dropIndex)) {
            result.column_names.push_back(df.column_names[i]);
            result.column_type[df.column_names[i]] = df.column_type[df.column_names[i]];
        }
    }
    
    // Iterate through each DataRow, removing the data at the dropIndex
    for (size_t i = 0; i < df.data.size(); ++i) {
        DataRow newRow;
        for (size_t j = 0; j < df.data[i].size(); ++j) {
            if (j != static_cast<size_t>(dropIndex)) {
                // Copy StockData with proper management of string type
                if (df.data[i][j].double_or_string) {
                    // If it's a string, use the string constructor
                    newRow.push_back(StockData(df.data[i][j].str));
                } else {
                    // If it's a double, use the double constructor
                    newRow.push_back(StockData(df.data[i][j].value));
                }
            }
        }
        result.data.push_back(newRow);
    }
    
    return result;
}



//implement the function of 'pandas.merge()' in python
DataFrame mergeColumn(DataFrame& A, DataFrame& B, vector<string> on_column_names) {
    DataFrame result;
    // Copy column names from A and then add unique column names from B
    result.column_names = A.column_names;
    result.column_type = A.column_type;
    for (const auto& columnName : B.column_names) {
        if (find(result.column_names.begin(), result.column_names.end(), columnName) == result.column_names.end()) {
            result.column_names.push_back(columnName);
            
            bool coltype = B.column_type[columnName];
            result.column_type[columnName] = coltype;
        }
    }
    
    // Find indices of key columns in both DataFrames
    vector<int> aIndices, bIndices;
    for (const auto& name : on_column_names) {
        aIndices.push_back(getColumnIdx(A, name));
        bIndices.push_back(getColumnIdx(B, name));
    }
    
    for (const auto& aRow : A.data) {
        DataRow newRow = createRow(result);
        
        bool matchFound = false;
        
        // Copy A row data to new row
        for (size_t i = 0; i < A.column_names.size(); ++i)
            newRow[i] = aRow[i];
        
        
        // Attempt to find matching row in B
        for (const auto& bRow : B.data) {
            bool isMatch = true;
            for (size_t i = 0; i < aIndices.size(); ++i) {
                if (aRow[aIndices[i]].str != bRow[bIndices[i]].str) {
                    isMatch = false;
                    break;
                }
            }
            
            if (isMatch) {
                // Found match, copy B row data to new row
                for (size_t i = 0; i < B.column_names.size(); ++i) {
                    int indexInResult = getColumnIdx(result, B.column_names[i]);
                    newRow[indexInResult] = bRow[i];
                }
                matchFound = true;
                break;
            }
        }
        
        if (!matchFound) {
            // If no match is found, fill B columns in newRow with default StockData instances
            for (size_t i = A.column_names.size(); i < result.column_names.size(); ++i) {
                string colname = result.column_names[i];
                bool coltype = getColumnType(result, colname);
                if(coltype) {
                    newRow[i] = StockData(string());
                }
                else {
                    newRow[i] = StockData(NAN);
                }
            }
        }
        
        result.data.push_back(newRow);
    }
    
    return result;
}



DataFrame fill_weight(DataFrame& df, string column) {
    int columnIndex = getColumnIdx(df, column);
    if (columnIndex == -1) return df; // Column not found
    
    for (int i = (int)df.data.size() - 1; i >= 0; --i) {
        // Check if the current row's value in the specified column is not NA
        if (!df.data[i][columnIndex].double_or_string && !isnan(df.data[i][columnIndex].value)) {
            double valueToFill = df.data[i][columnIndex].value;
            // Fill forward for up to 20 rows or until the next non-NA value
            for (size_t j = i + 1; j < i + 20 && j < df.data.size(); ++j) {
                if (!df.data[j][columnIndex].double_or_string && isnan(df.data[j][columnIndex].value)) {
                    df.data[j][columnIndex].value = valueToFill;
                } else {
                    // If the next value is not NA, stop filling forward
                    break;
                }
            }
        }
    }
    
    return df;
}


DataFrame calcReturn(DataFrame& df) {
    int idxDate = getColumnIdx(df, "date");
    int idxReturn = getColumnIdx(df, "return");
    int idxWeightX = getColumnIdx(df, "weight_x");
    int idxWeightY = getColumnIdx(df, "weight_y");
    
    double dailyTotalReturn = 0.0;
    string date;
    
    for (size_t i = 0; i < df.data.size(); ++i) {
        double ret = df.data[i][idxReturn].value;
        double weightX = df.data[i][idxWeightX].value;
        double weightY = df.data[i][idxWeightY].value;
        dailyTotalReturn += ret * (weightX - weightY);
        date = df.data[i][idxDate].str;
    }
    
    
    DataFrame res;
    res.column_names.push_back("date");
    res.column_names.push_back("daily_total_return");
    res.column_type["daily_total_return"] = false;
    res.column_type["date"] = true;
    
    StockData item1(date);
    StockData item2(dailyTotalReturn);
    DataRow row;
    row.push_back(item1);
    row.push_back(item2);
    res.data.push_back(row);
    
    return res;
}


//combine a set of DataFrame into a single one
DataFrame combine(const vector<DataFrame>& groups, DataFrame& df) {
    DataFrame combined;
    if (groups.empty()) return DataFrame(df);
    
    // Initialize column names from the first group
    combined.column_names = groups[0].column_names;
    combined.column_type = groups[0].column_type;
    // Iterate through each group and append its rows to the combined DataFrame
    for (size_t i = 0; i < groups.size(); ++i) {
        combined.data.insert(combined.data.end(), groups[i].data.begin(), groups[i].data.end());
    }
    
    return combined;
}

/*
 
 ################################################################################################
 '''Multi-Factor Portfolio Backtesting Simulation'''
 ################################################################################################
 
 */

int main(int argc, const char * argv[]) {
    
    //printf("%d %d", isnan(NAN), isnan(0));
    //return 0;
    //initdata
    
    srand(static_cast<unsigned int>(time(0))); // Seed random number generator
    
    DataFrame df;
    df.column_names = vector<string>{"stock_id", "date", "return", "factor", "market_value", "industry"};
    vector<bool> column_type{true, true, false, false, false, true};
    for(size_t i = 0; i < column_type.size(); ++i) {
        df.column_type[df.column_names[i]] = column_type[i];
    }
    const int numDays = 366; // Leap year
    const int numStocks = 10;
    const char* stockIds[] = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"};
    const char* industries[] = {"Tech", "Tech", "Finance", "Finance", "Energy", "Energy", "Tech", "Finance", "Energy", "Tech"};
    string startDate = "2020-01-01";
    
    //    int day = date2day(startDate);
    
    for (int day = 0; day < numDays; ++day) {
        
        time_t t = day * 24 * 3600;
        struct tm* tm = localtime(&t);
        char buf[100];
        strftime(buf, sizeof(buf), "%Y-%m-%d", tm);
        
        for (int stock = 0; stock < numStocks; ++stock) {
            DataRow row;
            row.push_back(StockData(stockIds[stock])); // stock_id
            // Calculate and add date
            row.push_back(StockData(buf)); // Placeholder for date, should calculate actual date
            row.push_back(StockData(static_cast<double>(rand()) / RAND_MAX - 0.5)); // return
            row.push_back(StockData(static_cast<double>(rand()) / RAND_MAX)); // factor
            row.push_back(StockData(static_cast<double>(rand()) / RAND_MAX * 10000)); // market_value
            row.push_back(StockData(industries[stock])); // industry
            df.data.push_back(row);
        }
    }
    
    //    DataFrame
    df = readCSV("/Users/wenbogeng/Documents/XcodeProjects/ruida/ruida/my_dataframe.csv");
    
    print(df, -1, "df block-2");
    
    
    // Group by "date" and apply 'winsorize'
    vector<DataFrame> groups = groupBy(df, "date");
    for (size_t i = 0; i < groups.size(); ++i) {
        winsorize(groups[i]);
    }
    // Combine groups back into a single DataFrame, simulating 'reset_index(drop=True)'
    DataFrame df_grouped;
    for (size_t i = 0; i < groups.size(); ++i) {
        df_grouped.data.insert(df_grouped.data.end(), groups[i].data.begin(), groups[i].data.end());
    }
    df_grouped.column_names = groups[0].column_names;
    df_grouped.column_type = groups[0].column_type;
    
    print(df_grouped, -1, "df_grouped block-7");
    
    
    // Group by "date" again and apply 'standardize'
    groups = groupBy(df_grouped, "date");
    for (size_t i = 0; i < groups.size(); ++i) {
        standardize(groups[i]);
    }
    // Combine groups into a single DataFrame for standardization
    DataFrame df_standardized;
    for (size_t i = 0; i < groups.size(); ++i) {
        df_standardized.data.insert(df_standardized.data.end(), groups[i].data.begin(), groups[i].data.end());
    }
    df_standardized.column_names = groups[0].column_names;
    df_standardized.column_type = groups[0].column_type;
    
    print(df_standardized , -1, "df_standardized block-8");
    
    // Group by "date" again and apply 'neutralize'
    groups = groupBy(df_standardized, "date");
    for (size_t i = 0; i < groups.size(); ++i) {
        neutralize(groups[i]);
    }
    // Combine groups into a single DataFrame for neutralization
    DataFrame df_neutralize(groups[0]);
    
    for (size_t i = 0; i < groups.size(); ++i) {
        df_neutralize.data.insert(df_neutralize.data.end(), groups[i].data.begin(), groups[i].data.end());
    }
    df_neutralize.column_names = groups[0].column_names;
    df_neutralize.column_type = groups[0].column_type;
    print(df_neutralize , -1, "df_neutralize block-10");
    
    int group_num = 5;
    int interval = 20;
    
    // Group by "stock_id"
    groups = groupBy(df_neutralize, "stock_id");
    
    // Resample and select the first entry for each group
    vector<DataFrame> resampledGroups;
    for (size_t i = 0; i < groups.size(); ++i) {
        vector<DataFrame> resampled = resample(groups[i], interval);
        for (size_t j = 0; j < resampled.size(); ++j) {
            if (!resampled[j].data.empty()) {
                DataFrame firstEntry;
                firstEntry.column_names = resampled[j].column_names;
                firstEntry.column_type = resampled[j].column_type;
                firstEntry.data.push_back(resampled[j].data.front());
                resampledGroups.push_back(firstEntry);
            }
        }
    }
    
    /*
     print(resampledGroups[0], -1, "resampledGroups");
     print(resampledGroups[1], -1, "resampledGroups");
     print(resampledGroups[2], -1, "resampledGroups");
     
     print(groups[0], -1, "groups[0]");
     */
    
    // Combine all first entries into a single DataFrame, simulating the 'first' operation after resampling
    DataFrame df_monthly;
    for (size_t i = 0; i < resampledGroups.size(); ++i) {
        df_monthly.data.insert(df_monthly.data.end(), resampledGroups[i].data.begin(), resampledGroups[i].data.end());
    }
    if (!resampledGroups.empty()) {
        df_monthly.column_names = resampledGroups[0].column_names;
        df_monthly.column_type = resampledGroups[0].column_type;
    }
    
    
    //    print(df_monthly, -1, "df_monthly block-16");
    
    
    group_num = 5;
    
    // Group by "date"
    groups = groupBy(df_monthly, "date");
    
    for (size_t i = 0; i < groups.size(); ++i) {
        // Extract 'factor_n' column values
        vector<double> factor_n_values;
        int factor_n_idx = getColumnIdx(groups[i], "factor_n");
        for (size_t j = 0; j < groups[i].data.size(); ++j) {
            if (!groups[i].data[j][factor_n_idx].double_or_string) { // Ensure it's a double
                factor_n_values.push_back(groups[i].data[j][factor_n_idx].value);
            }
        }
        
        // Calculate quantiles and assign them
        vector<int> quantile_assignments = qcut(factor_n_values, group_num);
        
        // Add or update the 'quantile' column in the original DataFrame group
        int quantile_col_idx = getColumnIdx(groups[i], "quantile");
        bool quantile_col_exists = (quantile_col_idx != -1);
        
        for (size_t j = 0; j < groups[i].data.size(); ++j) {
            StockData quantileData(0.0);
            quantileData.double_or_string = false; // Using double type for integer values, due to the structure definition
            quantileData.value = static_cast<double>(quantile_assignments[j]);
            
            if (quantile_col_exists) {
                groups[i].data[j][quantile_col_idx] = quantileData;
            } else {
                groups[i].data[j].push_back(quantileData);
            }
        }
        
        if (!quantile_col_exists) {
            groups[i].column_names.push_back("quantile");
            groups[i].column_type["quantile"] = false;
        }
    }
    
    df_monthly = combine(groups, df_monthly);
    
    groups = groupBy(df_monthly, "stock_id");
    df_monthly = combine(groups, df_monthly);
    
    print(df_monthly , -1, "df_monthly_quantile block-16");
    
    group_num = 5;
    
    // Split df_monthly into df_high and df_low based on quantile values
    DataFrame df_high(df_monthly, false), df_low(df_monthly, false);
    
    int quantileIdx = getColumnIdx(df_monthly, "quantile");
    int marketValueIdx = getColumnIdx(df_monthly, "market_value");
    
    // Filtering rows for high and low DataFrames
    for (size_t i = 0; i < df_monthly.data.size(); ++i) {
        if (compfloat(static_cast<int>(df_monthly.data[i][quantileIdx].value), group_num)) {
            df_high.data.push_back(df_monthly.data[i]);
        } else if (compfloat(static_cast<int>(df_monthly.data[i][quantileIdx].value), 0)) {
            df_low.data.push_back(df_monthly.data[i]);
        }
    }
    
    
    print(df_high , -1, "df_high block-18");
    print(df_low , -1, "df_low block-19");
    
    
    // Calculate weights for df_high and df_low
    vector<DataFrame> highGroups = groupBy(df_high, "date");
    vector<DataFrame> lowGroups = groupBy(df_low, "date");
    
    for (size_t i = 0; i < highGroups.size(); ++i) {
        double sumMarketValue = 0;
        for (size_t j = 0; j < highGroups[i].data.size(); ++j) {
            sumMarketValue += highGroups[i].data[j][marketValueIdx].value;
        }
        for (size_t j = 0; j < highGroups[i].data.size(); ++j) {
            highGroups[i].data[j].push_back(StockData(highGroups[i].data[j][marketValueIdx].value / sumMarketValue));
        }
        highGroups[i].column_names.push_back("weight");
        highGroups[i].column_type["weight"] = false;
    }
    
    for (size_t i = 0; i < lowGroups.size(); ++i) {
        double sumMarketValue = 0;
        for (size_t j = 0; j < lowGroups[i].data.size(); ++j) {
            sumMarketValue += lowGroups[i].data[j][marketValueIdx].value;
        }
        for (size_t j = 0; j < lowGroups[i].data.size(); ++j) {
            lowGroups[i].data[j].push_back(StockData(lowGroups[i].data[j][marketValueIdx].value / sumMarketValue));
        }
        lowGroups[i].column_names.push_back("weight");
        lowGroups[i].column_type["weight"] = false;
    }
    
    // Reconstruct df_high and df_low from grouped data
    df_high = combine(highGroups, df_high);
    df_low = combine(lowGroups, df_low);
    
    groups = groupBy(df_high, "stock_id");
    df_high = combine(groups, df_high);
    groups = groupBy(df_low, "stock_id");
    df_low = combine(groups, df_low);
    
    
    print(df_high , -1, "df_high block-21");
    print(df_low , -1, "df_low block-22");
    
    
    
    // Drop "stock_id" column and reset index
    DataFrame df_t = df_high;
    DataFrame df_b = df_low;
    DataFrame df_merge = df;
    
    //print(df_merge, -1, "df_merge");
    
    // Prepare df_t and df_b for merging by ensuring only "date", "stock_id", and 'weight' columns are present
    df_t = selectColumns(df_t, vector<string>{"stock_id", "date", "weight"});
    df_b = selectColumns(df_b, vector<string>{"stock_id", "date", "weight"});
    
    // Rename 'weight' column in df_t and df_b to distinguish them in the merged DataFrame
    renameColumn(df_t, "weight", "weight_x");
    renameColumn(df_b, "weight", "weight_y");
    
    // Perform merges (assuming a mergeColumn function that can handle 'left' joins exists)
    DataFrame df_daily = mergeColumn(df_merge, df_t, vector<string>{"date", "stock_id"});
    df_daily = mergeColumn(df_daily, df_b, vector<string>{"date", "stock_id"});
    
    
    print(df_daily, -1, "df_daily block-27");
    
    DataFrame a = groupBy(df_daily, "date")[20];
    print(a, -1, "test");
    
    DataFrame df_reset_index = df_daily;
    
    // Group by "stock_id" and apply 'fill_weight' for 'weight_x'
    groups = groupBy(df_reset_index, "stock_id");
    for (size_t i = 0; i < groups.size(); ++i) {
        fill_weight(groups[i], "weight_x");
    }
    df_grouped = combine(groups, df_reset_index);
    
    print(df_grouped, -1, "df_grouped_fill_weight_x block-30");
    
    
    
    
    // Group by "stock_id" again and apply 'fill_weight' for 'weight_y'
    groups = groupBy(df_reset_index, "stock_id");
    for (size_t i = 0; i < groups.size(); ++i) {
        fill_weight(groups[i], "weight_y");
    }
    DataFrame df_grouped_low = combine(groups, df_reset_index);
    
    print(df_grouped_low, -1, "df_grouped_low_fill_weight_y block-31");
    
    
    
    // Extract 'weight_y' column from df_grouped_low
    vector<StockData> weightYColumn = selectColumn(df_grouped_low, "weight_y");
    
    // Add 'weight_y' column from df_grouped_low to df_grouped
    df_grouped = dropColumn(df_grouped, "weight_y");
    df_grouped = addColumn(df_grouped, "weight_y", weightYColumn, false);
    
    // Ensure 'weight_x' and 'weight_y' are filled with 0 where they are NA
    int weightXIndex = getColumnIdx(df_grouped, "weight_x");
    int weightYIndex = getColumnIdx(df_grouped, "weight_y");
    
    for (size_t i = 0; i < df_grouped.data.size(); ++i) {
        if (isnan(df_grouped.data[i][weightXIndex].value)) {
            df_grouped.data[i][weightXIndex].value = 0.0;
        }
        if (isnan(df_grouped.data[i][weightYIndex].value)) {
            df_grouped.data[i][weightYIndex].value = 0.0;
        }
    }
    
    
    print(df_grouped, -1, "df_grouped block-33");
    
    
    // Group by "date"
    int dateIdx = getColumnIdx(df_grouped, "date");
    groups = groupBy(df_grouped, "date");
    
    DataFrame daily_returns;
    daily_returns.column_names.push_back("date");
    daily_returns.column_names.push_back("daily_total_return");
    daily_returns.column_type["date"] = true;
    daily_returns.column_type["daily_total_return"] = false;
    
    for (size_t i = 0; i < groups.size(); ++i) {
        double totalReturn = 0.0;
        int returnIndex = getColumnIdx(groups[i], "return");
        int weightXIndex = getColumnIdx(groups[i], "weight_x");
        int weightYIndex = getColumnIdx(groups[i], "weight_y");
        
        for (size_t j = 0; j < groups[i].data.size(); ++j) {
            double returnVal = groups[i].data[j][returnIndex].value;
            double weightXVal = groups[i].data[j][weightXIndex].value;
            double weightYVal = groups[i].data[j][weightYIndex].value;
            totalReturn += returnVal * (weightXVal - weightYVal);
        }
        
        DataRow newRow;
        newRow.push_back(groups[i].data.front()[dateIdx]); // Copy the date from the first row of the group
        newRow.push_back(StockData(totalReturn)); // Add the calculated total return
        daily_returns.data.push_back(newRow);
    }
    
    
    print(daily_returns, -1, "daily_returns block-34");
    
    
    double sum = 0.0;
    double sumSquared = 0.0;
    int totalReturnsIndex = getColumnIdx(daily_returns, "daily_total_return");
    size_t n = daily_returns.data.size();
    
    for (size_t i = 0; i < n; ++i) {
        double val = daily_returns.data[i][totalReturnsIndex].value;
        sum += val;
        sumSquared += val * val;
    }
    
    double mean = sum / n;
    double variance = (sumSquared / n) - (mean * mean);
    double stdDeviation = sqrt(variance);
    
    double annualizedReturn = mean * 252;
    double annualizedVol = stdDeviation * sqrt(252);
    double sharpeRatio = annualizedReturn / annualizedVol;
    
    printf("annualizedReturn=%f\nannualizedVol=%f\nsharpeRatio=%f\n", annualizedReturn, annualizedVol, sharpeRatio);
    return 0;
}

