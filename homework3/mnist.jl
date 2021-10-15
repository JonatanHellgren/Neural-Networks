using MAT

function load_mat_data()
  file = matopen("xTest2.mat")
  xTest = read(file)
  xTest = xTest["xTest2"]
end


