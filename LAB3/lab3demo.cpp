#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <random>

#define _USE_MATH_DEFINES

#include "mpi.h"
#include "math.h"


struct Body {
	double m;	 
	double r[3]; 
	double v[3];
};

std::ostream& operator<<(std::ostream& str, const Body& b) {
	str << std::setprecision(10) << b.r[0] << " " << b.r[1] << " " << b.r[2] << std::endl;

	return str;
}

double norm_vec(const double* r) {
	return sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
}

double max(double a, double b){
	return ((a) > (b)) ? (a) : (b);
}

double power3(double a) {
	return a * a * a;
}

void acceleration(double* a, const size_t N, const double* r,  const Body* data, const double G) {

	double buf[3] = { 0.0 , 0.0, 0.0 };

	for (size_t k = 0; k < 3; ++k)
		a[k] = 0.0;

	double coeff = 1.0;

	Body bod_j;

	for (size_t j = 0; j < N; ++j) {

		bod_j = data[j];

		for (size_t k = 0; k < 3; ++k)
			buf[k] = bod_j.r[k] - r[k];

		coeff = power3(1.0 / max(norm_vec(buf), 1e-6));

		for (size_t k = 0; k < 3; ++k) {
			buf[k] *= G * bod_j.m * coeff;
			a[k] += buf[k];
		}
	}
}

void read_file(const std::string& file_name, Body* data, size_t& N)
{
	std::ifstream F(file_name);

	F >> N;
	
	for (size_t i = 0; i < N; ++i)
		F >> data[i].m >> data[i].r[0] >> data[i].r[1] >> data[i].r[2] >> data[i].v[0] >> data[i].v[1] >> data[i].v[2];

	F.close();
}

int main(int argc, char** argv) {
	size_t N = 10000; // Количество тел

	int num = 4; // Номер разбиения

	double tau = num * 1e-3; // Шаг по времени
	double timeStep = 1e-1;  // Шаг записи в файлы

	double tEnd = 20.0; // Конечный момент времени

	size_t Nt = round(tEnd / tau);       // Количество шагов по времени
	size_t tf = round(timeStep / tau); // Коэффициент пропорциональности шагов

	int flag1 = 0;        // != 0 - считывать из файла, 0 - заполнять случайно
	int flag2 = 0;        // != 0 - записывать в файлы, 0 - нет

	double mLeft = 1e+9;  
	double mRight = 1e+10;  

	double rLeft = -1.0;  
	double rRight = 1.0;   

	double vLeft = -1.0;  
	double vRight = 1.0; 

	std::random_device rd; 
	std::mt19937 gen(rd());

	std::uniform_real_distribution<double> dis(0.0, 1.0);


	const double G = 6.67e-11;

	int rank = 0; // Номер текущего процесса
	int np = 0; // Общее число всех процессов

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	Body* data = NULL; // Массив "тел"

	int* locSize = new int[np];	// Массивы длин и смещений
	int* locOffset = new int[np]; // локальных массивов

	if (rank == 0) {
		if (flag1) {
			data = new Body[N];
			read_file("4body.txt", data, N);
		}
		else {
			data = new Body[N]; 

			for (size_t i = 0; i < N; ++i) {
				data[i].m = mLeft + dis(gen) * (mRight - mLeft);

				for (size_t k = 0; k < 3; ++k) {
					data[i].r[k] = rLeft + dis(gen) * (rRight - rLeft);
					data[i].v[k] = vLeft + dis(gen) * (vRight - vLeft);
				}
			}
		}

		locOffset[0] = 0;

		int L = N / np; // Размер частей массива

		for (int p = 0; p < np - 1; ++p) {
			locSize[p] = L;
			locOffset[p + 1] = locOffset[p] + L;
		}

		locSize[np - 1] = L + N % np;
	}

	MPI_Bcast(locSize, np, MPI_INT, 0, MPI_COMM_WORLD);   // Рассылка всем
	MPI_Bcast(locOffset, np, MPI_INT, 0, MPI_COMM_WORLD); // процессам

	Body* locDat = new Body[locSize[rank]]; // Локальный массив "тел"

	const int nItems = 3; 

	int blockLengths[3] = { 1, 3, 3 }; 

	MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE }; 

	MPI_Aint offsets[3] = { offsetof(Body, m), offsetof(Body, r), offsetof(Body, v) };

	MPI_Datatype mpi_Body; // Новый тип данных - структура "тело"

	MPI_Type_create_struct(nItems, blockLengths, offsets, types, &mpi_Body);

	MPI_Type_commit(&mpi_Body);

	// Тип только для позиций
	int countR = 1;  
	int lengthsR[] = { 3 };  
	MPI_Aint offsetsR[] = { offsetof(Body, r) }; 
	MPI_Datatype typesR[] = { MPI_DOUBLE }; 

	MPI_Datatype mpi_tmp; // Вспомогательный тип
	MPI_Type_create_struct(countR, lengthsR, offsetsR, typesR, &mpi_tmp);

	MPI_Datatype mpi_Body_r;
	MPI_Type_create_resized(mpi_tmp, 0, 56, &mpi_Body_r); // Body состоит из 7 double => 7 x 8 = 56

	MPI_Type_commit(&mpi_Body_r);


	// Рассылка всем процессам частей массива "тел"
	MPI_Scatterv(data, locSize, locOffset, mpi_Body, locDat, locSize[rank], mpi_Body, 0, MPI_COMM_WORLD);

	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); // Рассылка числа тел всем процессам

	if (rank)
		data = new Body[N]; // Создание массива "тел" для остальных процессов

	MPI_Bcast(data, N, mpi_Body, 0, MPI_COMM_WORLD); // Рассылка массива тел всем процессам

	MPI_Bcast(&flag2, 1, MPI_INT, 0, MPI_COMM_WORLD); // Рассылка флага записи в файлы всем процессам

	std::ofstream* F = NULL;

	if (flag2) {
		F = new std::ofstream[locSize[rank]]; // Массив файлов для каждого узла

		for (size_t i = 0; i < locSize[rank]; ++i) {
			F[i].open(std::to_string(num) + "_Body_" + std::to_string(locOffset[rank] + i + 1) + ".txt");

			 //F[i].open("Body_" + std::to_string(locOffset[rank] + i + 1) + ".txt");

			//F[i] << N << " " << locDat[i].m << " " << tau << std::endl;

			F[i] << 0.0 << " " << locDat[i];
		}
	}

	Body bod_i; // Текущее тело


	double a[3] = { 0.0, 0.0, 0.0 }; // Текущие ускорения

	double start = 0.0;
	double end = 0.0;

	Body* locDatBuf = new Body[locSize[rank]];  // Промежуточный локальный массив "тел"
	double* w = new double[3 * locSize[rank]];  // Начальные ускорения

	start = MPI_Wtime();

	// Расчётная схема
	for (size_t t = 1; t <= Nt; ++t) {
		for (size_t i = 0; i < locSize[rank]; ++i) {
			bod_i = locDat[i];

			locDatBuf[i] = bod_i;

			acceleration(a, N, bod_i.r, data, G);

			for (size_t k = 0; k < 3; ++k) {
				locDatBuf[i].r[k] += tau * bod_i.v[k];

				locDatBuf[i].v[k] += tau * a[k];

				w[3 * i + k] = a[k];
			}
		}

		MPI_Allgatherv(locDatBuf, locSize[rank], mpi_Body_r, data, locSize, locOffset, mpi_Body_r, MPI_COMM_WORLD);

		for (size_t i = 0; i < locSize[rank]; ++i) {

			acceleration(a, N, locDatBuf[i].r, data, G);

			for (size_t k = 0; k < 3; ++k) {
				locDat[i].r[k] += 0.5* tau * (locDat[i].v[k] + locDatBuf[i].v[k]);
				locDat[i].v[k] += 0.5 * tau * (w[3 * i + k] + a[k]);
			}

			if (t % tf == 0 && flag2)
				F[i] << t * tau << " " << locDat[i];
		}

		MPI_Allgatherv(locDat, locSize[rank], mpi_Body, data, locSize, locOffset, mpi_Body, MPI_COMM_WORLD);
	}

	end = MPI_Wtime();

	if (flag2)
		for (size_t i = 0; i < locSize[rank]; ++i)
			F[i].close();

	if (rank == 0)
		std::cout << "time = " << end - start << std::endl << std::endl;

	delete[] F;
	delete[] data;
	delete[] locDat;
	delete[] locSize;
	delete[] locOffset;

	MPI_Finalize();

	return 0;
}
