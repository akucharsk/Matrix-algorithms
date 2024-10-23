csv_data = ["matrix_64.csv", "matrix_128.csv", "matrix_256.csv", "matrix_512.csv", "matrix_1024.csv", "matrix_2048.csv"];
times = [0, 0, 0, 0];

for i = 1:length(csv_data)
    matrix_data = csvread(csv_data(i));
    tic;
    matrix_data * matrix_data;
    time = toc;

    times(i) = time;

    str = sprintf("Time: %.8fs", time);
    disp(str);
end

csvwrite("results.csv", times);
