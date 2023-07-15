


#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <vector>
#include <tuple>
#include <functional>
#include <chrono>		// timer operation
#include <cstdio>		// remove file operation
#include <random>		// generate random numbers


#define MATRIX_INPUT_FILE_NAME	"data_8.txt"
#define PI						3.14159265358979323846
#define JACOBI_ERROR			1.0e-9


using namespace std;
using time_type = std::chrono::time_point<std::chrono::steady_clock>;


/* Timer utils */
time_type start_timer();

void end_timer(time_type start);


/* Math utils functions */
double vec_length(vector<double>& v);

void transpose(vector<vector<double>>& matrix);

vector<vector<double>> dot_product(vector<vector<double>>& matrix1, vector<vector<double>> matrix2);

double dot_product(vector<double>& v1, vector<double>& v2);

vector<double> dot_product(double scalar, vector<double>& v);

vector<vector<double>> rotation_product(vector<vector<double>>& matrix, vector<vector<double>>& rotation, int p, int q);

vector<vector<double>> diagonal_multiplication(vector<vector<double>>& matrix, vector<vector<double>>& diagonal);

vector<vector<double>> eye_matrix(int n);

vector<double> get_diag(vector<vector<double>>& matrix);

vector<vector<double>> diag_to_matrix(vector<double>& diagonal, int size);

vector<double> get_column(vector<vector<double>>& matrix, int column);


/* Format and print utils functions */
void print_vec(vector<double>& v);

void print_matrix(vector<vector<double>>& matrix);

vector<vector<double>> read_file(std::string file_name);

vector<double> read_line(std::string line);

bool write_to_file(std::string file_name, vector<vector<double>>& mtx);



/* Jacobi Eigenvalue Algorithm functions */
tuple<double, int, int> find_max_num(vector<vector<double>>& matrix);

tuple<vector<vector<double>>, double, double> calc_Jacobi_rotation_matrix(vector<vector<double>>& matrix, int p, int q);

void calc_matrix(vector<vector<double>>& mtx, double cos, double sin, int i, int j);

bool check_matrix_diagonality_and_update_zeros(vector<vector<double>>& matrix);

std::tuple<vector<vector<double>>, vector<double>, int> Jacobi(vector<vector<double>> matrix);

vector<std::tuple<double, vector<double>>>generate_eigen_tuples_vector(vector<vector<double>>& eigenvectors, vector<double>& lamdas);


/* SVD functions */
std::tuple<vector<vector<double>>, vector<double>, vector<vector<double>>>SVD(vector<vector<double>>& input_matrix);

void check_decomposition(vector<vector<double>>& input_matrix, vector<vector<double>> U, vector<double> Sigma, vector<vector<double>> V_T);




int main() 
{
	vector<vector<double>> matrix;

	cout << "Reading Input Matrix...";
	time_type start_read_file_time = start_timer();
	matrix = read_file(MATRIX_INPUT_FILE_NAME);
	cout << " (" << matrix.size() << ", " << matrix.at(0).size() << ")";
	end_timer(start_read_file_time);


	cout << "Compute SVD...";
	time_type start_svd_time = start_timer();
	auto tuple1 = SVD(matrix);
	vector<vector<double>> U = std::get<0>(tuple1);
	vector<double> Sigma = std::get<1>(tuple1);
	vector<vector<double>> V_T = std::get<2>(tuple1);
	end_timer(start_svd_time);


	std::cout << "U = " << std::endl;
	print_matrix(U);
	std::cout << "S = " << std::endl;
	print_vec(Sigma);
	std::cout << "V.T = " << std::endl;
	print_matrix(V_T);


	check_decomposition(matrix, U, Sigma, V_T);
	
	return 0;
}


time_type start_timer()
{
	return chrono::high_resolution_clock::now();
}

void end_timer(time_type start)
{
	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::seconds>(end - start);
	cout << " (" << duration.count() << " sec)" << endl;
}



/*
	This method return the vetor length.
*/
double vec_length(vector<double>& v) 
{
	double sum = 0;
	for (double num : v) {
		sum += pow(num, 2);
	}

	return sqrt(sum);
}


/*
This method performe the matrix transpose operation.
*/
void transpose(vector<vector<double>>& matrix) 
{
	
	int row = matrix.size();
	int col = matrix[0].size();

	// pre-allocation of the vector size: 
	vector<double> v (row);
	vector<vector<double>> new_matrix (col, v);

	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			new_matrix[i][j] = matrix[j][i];
		}
	}

	matrix = new_matrix;
}


/*
	dot_product method perform matrix multiplication, and return the result.
*/
vector<vector<double>> dot_product(vector<vector<double>>& matrix1, vector<vector<double>> matrix2) 
{
	int row_matrix1 = matrix1.size();
	int col_matrix1 = matrix1[0].size();
	int row_matrix2 = matrix2.size();
	int col_matrix2 = matrix2[0].size();

	if (col_matrix1 != row_matrix2) 
	{
		std::string str = "Cann't multiply (" + std::to_string(row_matrix1) + ", " + std::to_string(col_matrix1) + ") by (" + std::to_string(row_matrix2) + ", " + std::to_string(col_matrix2) + ").";
		std::cout << str << std::endl;
		throw std::invalid_argument(str);
	}

	transpose(matrix2);
	int row_matrix2_T = matrix2.size();

	vector<double> v(row_matrix2_T);
	vector<vector<double>> new_matrix(row_matrix1, v);

	for (int i = 0; i < row_matrix1; i++)
	{
		for (int j = 0; j < row_matrix2_T; j++)
		{
			double num = dot_product(matrix1[i], matrix2[j]);
			new_matrix[i][j] = num;
		}
	}

	return new_matrix;
}


/*
	dot_product method return the result of the multiplication between vector1 and vector2.
*/
double dot_product(vector<double>& v1, vector<double>& v2) 
{
	int n;

	if ((n = v1.size()) != v2.size()) {
		std::string str = "vectors are not in the same size.";
		throw std::invalid_argument(str);
	}

	double sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += (v1[i] * v2[i]);
	}

	return sum;
}


/*
	dot_product method return the result of the multiplication between given scalar and vector.
*/
vector<double> dot_product(double scalar, vector<double>& v) 
{
	for (int i = 0; i < v.size(); i++)
	{
		v[i] = scalar * v[i];
	}

	return v;
}


/*
	rotation_product method execute matrix multiplication of regular matrix and rotation matrix.
*/
vector<vector<double>> rotation_product(vector<vector<double>>& matrix, vector<vector<double>>& rotation, int p, int q) 
{
	vector<vector<double>> matrix_copy = matrix;

	int row_matrix = matrix_copy.size();
	int col_matrix = matrix_copy[0].size();
	int row_rotation = rotation.size();
	int col_rotation = rotation[0].size();


	if (col_matrix != row_rotation) 
	{
		std::string str = "Cann't multiply (" + std::to_string(row_matrix) + ", " + std::to_string(col_matrix) + ") by (" + std::to_string(row_rotation) + ", " + std::to_string(col_rotation) + ").";
		throw std::invalid_argument(str);
	}

	transpose(rotation);

	// changes is only have to apply in 2 matrix's columns only - columns p and q.
	const int inner_iteration = 2;

	for (int i = 0; i < row_matrix; i++)
	{
		double index_ip;
		double index_iq;
		for (int j = 0; j < inner_iteration; j++)
		{
			if (j == 0)
				index_ip = dot_product(matrix_copy[i], rotation[p]);
			else
				index_iq = dot_product(matrix_copy[i], rotation[q]);
		}
		matrix_copy[i][p] = index_ip;
		matrix_copy[i][q] = index_iq;
	}
	return matrix_copy;
}


/*
	diagonal_multiplication method execute matrix multiplication of regular matrix
	and diagonal matrix.
*/
vector<vector<double>> diagonal_multiplication(vector<vector<double>>& matrix, vector<vector<double>>& diagonal) 
{
	int row_matrix = matrix.size();
	int col_matrix = matrix[0].size();
	int row_diagonal = diagonal.size();
	int col_diagonal = diagonal[0].size();


	if (col_matrix != row_diagonal) 
	{
		std::string str = "Cann't multiply (" + std::to_string(row_matrix) + ", " + std::to_string(col_matrix) + ") by (" + std::to_string(row_diagonal) + ", " + std::to_string(col_diagonal) + ").";
		throw std::invalid_argument(str);
	}

	for (int i = 0; i < row_matrix; i++)
	{
		for (int j = 0; j < row_diagonal; j++)
			matrix[i][j] = matrix[i][j] * diagonal[j][j];
	}
	return matrix;
}


/*
	eye_matrix method return the identity matrix of size n.
*/
vector<vector<double>> eye_matrix(int n) 
{
	//pre-allocation of the vector size:
	vector<double> v (n, 0);
	vector<vector<double>> I(n,v);

	for (int i = 0; i < n; i++)
		I[i][i] = 1;

	return I;
}


/*
	get_diag method return vector that contains the elements that on the matrix diagonal.
*/
vector<double> get_diag(vector<vector<double>>& matrix) 
{
	int row = matrix.size();
	int col = matrix[0].size();

	//Check that the given matrix is square matrix
	if (row != col) {
		std::string str = "The given matrix is not square matrix.";
		throw std::invalid_argument(str);
	}

	//pre-allocation of the vector size: 
	vector<double> diagonal (row);

	for (int i = 0; i < row; i++)
	{
		diagonal[i] = matrix[i][i];
	}
	return diagonal;
}


/*
	diag_to_matrix method convert a vector to diagonal matrix in size size_val value.
*/
vector<vector<double>> diag_to_matrix(vector<double>& diagonal, int size_val) 
{
	int n = diagonal.size();

	if (n > size_val) 
	{
		std::string str = "The desire matrix size is smaller than the amount of elements in diagonal vectors";
		throw std::invalid_argument(str);
	}

	vector<double> v(size_val, 0);
	vector<vector<double>> new_matrix(size_val, v);

	for (int i = 0; i < n; i++)
	{
		new_matrix[i][i] = diagonal[i];
	}

	return new_matrix;
}


/*
	get_column method get matrix and column. The method rturn vector 
	with the elements of this matrix's column.
*/
vector<double> get_column(vector<vector<double>>& matrix, int column) 
{
	if (matrix[0].size() < column) 
	{
		std::string str = "The given matrix do not have column number " + std::to_string(column) + ".";
		throw std::invalid_argument(str);
	}
	
	int n = matrix.size();
	vector<double> v(n);

	for (int i = 0; i < n; i++)
	{
		v[i] = matrix[i][column];
	}
	return v;
}


/*
	print_vec method print the given vector to the console.
*/
void print_vec(vector<double>& v) 
{
	bool flag = false;

	for (int i = 0; i < v.size(); i++)
	{
		if (v.size() == 1)
			std::cout << "[" << v[i] << "]" << std::endl;
		else if (i == 0)
			std::cout << "[" << v[i] << ",	";
		else if (i + 1 == v.size())
			std::cout << v[i] << "]" << std::endl;
		else {
			std::cout << v[i] << ",	";

			//The case the given vector is bigger then 10 elements:
			if (i > 2 && v.size() > 10 && !flag) {
				std::cout << "...	";
				i = v.size() - 4;
				flag = true;
			}
		}
	}
}


/*
	print_matrix method print the given matrix to the console.
*/
void print_matrix(vector<vector<double>>& matrix) 
{
	bool flag = false;

	for (int i = 0; i < matrix.size(); i++)
	{
		if (matrix.size() == 1) {
			std::cout << "[\n ";
			print_vec(matrix[i]);
			std::cout << "]" << std::endl;
		}
		else if (i == 0) {
			std::cout << "[ ";
			print_vec(matrix[i]);
		}
		else if (i + 1 == matrix.size()) {
			std::cout << " ";
			print_vec(matrix[i]);
			std::cout << "] " << std::endl;
		}
		else {
			std::cout << " ";
			print_vec(matrix[i]);

			//The case the given matrix is bigger then 10 elements:
			if (i > 2 && matrix.size() > 10 && !flag) {
				std::cout << " ...	\n";
				i = matrix.size() - 4;
				flag = true;
			}
		}
	}
}


/*
	read_file method read the all content from the input file.
	The method return matrix with all the values from the input file.
*/
vector<vector<double>> read_file(std::string file_name) 
{
	std::string line;
	fstream newfile;
	vector<vector<double>> matrix;

	
	//ios::in - is mode that represent the operation of open for input operations.
	newfile.open(file_name, ios::in); //open a file to perform read operation using file object
	if (!newfile.good()) {
		cout << "Error: failed to read the input matrix" << endl;
		exit(1);
	}
	if (newfile.is_open()) {   //checking that the file is open

		while (std::getline(newfile, line)) { //read data from file object and put it into string.
			vector<double> vector = read_line(line);
			matrix.push_back(vector);
		}

		newfile.close(); //close the file object.
	}
	return matrix;
}

/*
	read_line method get string value and convert it to vector of double.
*/
vector<double> read_line(std::string line) 
{
	vector<double> v;

	size_t pos = 0;
	std::string delimiter = ",";

	while ((pos = line.find(delimiter)) != std::string::npos) {
		std::string token = line.substr(0, pos);

		double num = std::stod((const std::string&)token);
		v.push_back(num);

		line.erase(0, pos + delimiter.length());
	}

	//Last number in the line do not have comma after it.
	double num = std::stod((const std::string&)line);
	v.push_back(num);

	return v;
}




/*
	write_to_file method write the content of the given matrix to the file. Returns True if succeeded.
*/
bool write_to_file(std::string file_name, vector<vector<double>>& mtx) 
{
	int row = mtx.size();
	int col = mtx[0].size();

	std::remove(file_name.c_str());

	ofstream myfile;
	// out		= opens the file for writing
	// app		= append
	// binary	= makes sure the data is read or written without translating new line characters to and from \r\n on the fly
	myfile.open(file_name, ios::out | ios::app | ios::binary);

	if (myfile.is_open()) 
	{

		std::string delimiter = ",";
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				// Case of the last number in the matrix
				if (!(j + 1 < col))		
					myfile << mtx[i][j];
				else
					myfile << mtx[i][j] << ",";
			}
			if (i + 1 < row)
				myfile << "\n";
		}
		myfile.close();
		return true;
	}
	return false;
}


/*
	find_max_num method find the largest absolute value off-diagonal element, 
	and return his indexes and value.
	The given matrix is symmetric, so only need to search in the upper triangular matrix.
*/
std::tuple<double, int, int> find_max_num(vector<vector<double>>& matrix) 
{
	int row = matrix.size();
	int col = matrix[0].size();
	int p, q;
	double max_val;

	for (int i = 0; i < row; i++)
	{
		for (int j = i+1; j < col; j++)
		{
			if (i == 0 && j == 1) {
				max_val = std::abs(matrix[i][j]);
				p = i;
				q = j;
			}
			else if (max_val < std::abs(matrix[i][j])) {
				max_val = std::abs(matrix[i][j]);
				p = i;
				q = j;
			}
		}
	}

	return std::make_tuple(max_val, p, q);
}


/*
	calce_J_matrix method find the rotation matrix that called J:
*/
std::tuple<vector<vector<double>>, double, double>calc_Jacobi_rotation_matrix(vector<vector<double>>& matrix, int p, int q)
{
	//Check that the given matrix is square matrix
	if (matrix.size() != matrix[0].size()) {
		std::string str = "The given matrix is not square matrix.";
		throw std::invalid_argument(str);
	}

	//alocation new identity matrix:
	int n = matrix.size();
	vector<vector<double>> J = eye_matrix(n);

	double theta;

	//calculate theta:
	if (matrix[q][q] == matrix[p][p])
		theta = PI / 4;
	else {
		double a = (2 * matrix[p][q]) / (matrix[q][q] - matrix[p][p]);
		theta = 0.5 * atan(a);
	}

	double cosinus, sinus;
	//insert new values to different places in the matrix J :
	J[p][p] = J[q][q] = cosinus = cos(theta);
	J[q][p] = sinus = sin(theta);
	J[p][q] = -1 * sin(theta);


	return std::make_tuple(J, cosinus, sinus);
}


/*
	my_round check if num is smaller than JACOBI_ERROR value. If so, then num become zero.
*/
void my_round(double& num) 
{
	if (std::abs(num) < JACOBI_ERROR)
		num = 0;
}


/*
	For make the performention of Jacobi Eigenvalue Algorithm better,
	matrix multiplication of J.T*A*J is replaced by some elementary operations.
	And then, the all function is O(n) instead of O(n^2).
*/
void calc_matrix(vector<vector<double>>& mtx, double cos, double sin, int i, int j) 
{
	double a_ii = mtx[i][i];
	double a_ij = mtx[i][j];
	double a_jj = mtx[j][j];
	double a_ji = mtx[j][i];

	mtx[i][i] = pow(cos, 2) * a_ii - 2 * sin * cos * a_ij + pow(sin, 2) * a_jj;
	mtx[j][j] = pow(sin, 2) * a_ii + 2 * sin * cos * a_ij + pow(cos, 2) * a_jj;
	mtx[i][j] = mtx[j][i] = (pow(cos, 2) - pow(sin, 2)) * a_ij + sin * cos * (a_ii - a_jj);


	for (int k = 0; k < mtx.size(); k++)
	{
		if (k != i && k != j) {
			double a_ik = mtx[i][k];
			double a_jk = mtx[j][k];
			mtx[i][k] = mtx[k][i] = cos * a_ik - sin * a_jk;
			mtx[j][k] = mtx[k][j] = sin * a_ik + cos * a_jk;
		}
	}
}



/*
	check_and_update method is doing the following things:
	Convert matrix elemnt that is smaller than JACOBI_ERROR (almost equal to 0) to be 0.
	Check if the given matrix is diagonal.
*/
bool check_matrix_diagonality_and_update_zeros(vector<vector<double>>& matrix)
{
	int rows_num = matrix.size();
	int cols_num = matrix[0].size();

	//Check that the given matrix is square matrix
	if (rows_num != cols_num) {
		std::string str = "The given matrix is not square matrix.";
		throw std::invalid_argument(str);
	}

	for (int i = 0; i < rows_num; i++)
	{
		for (int j = i+1; j < cols_num; j++)
		{
			if (abs(matrix[i][j]) < JACOBI_ERROR)
				matrix[i][j] = matrix[j][i] = 0;
			else
				return false;
		}
	}
	return true;
}


/*
	Jacobi method Implemntion
	Jacobi Eigenvalue Algorithm: https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
	The method return Eigenvalues and Eigenvectors.
*/

std::tuple<vector<vector<double>>, vector<double>, int>Jacobi(vector<vector<double>> matrix) 
{
	//Check that the given matrix is square matrix
	if (matrix.size() != matrix[0].size()) {
		std::string str = "The given matrix is not square matrix.";
		throw std::invalid_argument(str);
	}

	//Initialize of the variables:
	int n = matrix.size();
	vector<vector<double>> J = eye_matrix(n);

	//Set limit on the number of iterations:
	int max_iterations = 100;
	int cur_iteration_num = 0;

	for (int i = 0; i < max_iterations; i++)
	{
		//Get matrix max number and his index:
		auto tuple1 = find_max_num(matrix);
		double max_val = std::get<0>(tuple1);
		int p = std::get<1>(tuple1);
		int q = std::get<2>(tuple1);

		if (max_val < JACOBI_ERROR)
			return std::make_tuple(J, get_diag(matrix), cur_iteration_num);

		//Get rotation matrix and get cos and sin values:
		auto tuple2 = calc_Jacobi_rotation_matrix(matrix, p, q);
		vector<vector<double>> J1 = std::get<0>(tuple2);
		double cosinus = std::get<1>(tuple2);
		double sinus = std::get<2>(tuple2);

		//Calculate the new matrix:
		calc_matrix(matrix, cosinus, sinus, p, q);

		//Calculate the eigenvectors:
		J = rotation_product(J, J1, p, q);		

		cur_iteration_num++;
		if (check_matrix_diagonality_and_update_zeros(matrix))
			break;
	}

	return std::make_tuple(J, get_diag(matrix), cur_iteration_num);
}


/*
	Create an array of (Eigenvalue, Eigenvector) tupples,
	sorted in descending order according to the Eigenvalues.
	The array contains only tuples of positive Eigenvalues.
*/
vector<std::tuple<double, vector<double>>> generate_eigen_tuples_vector(vector<vector<double>>& eigenvectors, vector<double>& lamdas)
{
	//Initialize of the variables:
	vector<std::tuple<double, vector<double>>> t_vecs;
	bool flag = false;

	for (int i = 0; i < lamdas.size(); i++)
	{
		if (lamdas[i] > 0) {

			auto tuple = std::make_tuple(lamdas[i], get_column(eigenvectors, i));
			if (t_vecs.size() == 0)
				t_vecs.push_back(tuple);
			else {
				for (int j = 0; j < t_vecs.size(); j++)
				{
					auto tuple1 = t_vecs[j];
					if (std::get<0>(tuple1) <= lamdas[i]) {

						// Create Iterator pointing to the desire place:
						auto itPos = t_vecs.begin() + j;

						// Insert element to the desire position in vector:
						t_vecs.insert(itPos, tuple);
						flag = true;
						break;
					}
				}
				if (!flag)
					t_vecs.push_back(tuple);

				//reinisialize
				flag = false;
			}
		}
	}

	return t_vecs;
}


/*
	This method get input matrix and perfome Singular Value Decomposition (SVD).
*/
std::tuple<vector<vector<double>>, vector<double>, vector<vector<double>>> SVD(vector<vector<double>>& input_matrix) 
{	
	//copy constructor
	vector<vector<double>> AT = input_matrix;

	transpose(AT);

	vector<vector<double>> AT_A = dot_product(AT, input_matrix);

	auto tuple_Jacobi_result = Jacobi(AT_A);

	vector<vector<double>> eigenvectors = std::get<0>(tuple_Jacobi_result);
	vector<double> eigenvalues = std::get<1>(tuple_Jacobi_result);


	auto eigen_tuples_vec = generate_eigen_tuples_vector(eigenvectors, eigenvalues);

	// Sigma matrix - contain the Singular Values in descending order on the main diagonal
	vector<double> Sigma(eigen_tuples_vec.size());

	// U matrix - (1 / Singular Values) * A * V
	vector<vector<double>> U;

	// V.T matrix - contain the transpose of the eigenvectors.
	vector<vector<double>> V_T;

	for (int i = 0; i < eigen_tuples_vec.size(); i++)
	{
		auto curr_eigen_tuple = eigen_tuples_vec[i];

		double curr_eigenvalue = std::get<0>(curr_eigen_tuple);
		vector<double> curr_eigenvector = std::get<1>(curr_eigen_tuple);

		double s = sqrt(curr_eigenvalue);

		//Create v vector by normalize v1 vector:
		double vec_norm = vec_length(curr_eigenvector);
		vector<double> v = dot_product(1/vec_norm, curr_eigenvector);
		
		//Create u vector by multiply scalar on dot product of input matrix with v vector ==> (1/s)*input_matix*v:
		double scalar = 1 / s;
		vector<vector<double>> v_t = { v };
		transpose(v_t);

		v_t = dot_product(input_matrix, v_t);
		transpose(v_t);

		vector<double> u = dot_product(scalar, v_t[0]);
	
		//Insert values to the finale data structures:
		Sigma[i] = s;

		U.push_back(u);

		V_T.push_back(v);
	}

	transpose(U);

	return std::make_tuple(U, Sigma, V_T);
}


/*
	Checking that the U * S * VT is equal to input matrix.
*/
void check_decomposition(vector<vector<double>>& input_matrix, vector<vector<double>> U, vector<double> Sigma, vector<vector<double>> V_T) 
{
	std::cout << "Examination Of The Decomposition: " << std::endl;
	std::cout << "Inpute Matrix = " << std::endl;
	print_matrix(input_matrix);

	std::cout << std::endl;

	std::cout << "Result Of The 3 Matrix Multiplication From SVD Is:\nU*S*V.t =" << std::endl;

	vector<vector<double>> Sigma_matrix = diag_to_matrix(Sigma, U[0].size());

	vector<vector<double>> temp_mtx = diagonal_multiplication(U, Sigma_matrix);

	vector<vector<double>> res = dot_product(temp_mtx, V_T);

	print_matrix(res);
}