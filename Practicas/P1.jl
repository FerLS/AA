using FileIO;
using DelimitedFiles;
using Statistics;

# Cargamos el dataset
dataset = readdlm("iris.data", ',');

# Preparamos las entradas
inputs = dataset[:, 1:4];
# Con cualquiera de estas 3 maneras podemos convertir la matriz de entradas de tipo Array{Any,2} en Array{Float32,2}, si los valores son numéricos:
inputs = Float32.(inputs);
inputs = convert(Array{Float32,2}, inputs);
inputs = [Float32(x) for x in inputs];
println("Tamaño de la matriz de entradas: ", size(inputs, 1), "x", size(inputs, 2), " de tipo ", typeof(inputs));

# Preparamos las salidas deseadas codificándolas puesto que son categóricas
targets = dataset[:, 5];
println("Longitud del vector de salidas deseadas antes de codificar: ", length(targets), " de tipo ", typeof(targets));
classes = unique(targets);
numClasses = length(classes);
@assert(numClasses > 1);
if numClasses == 2
    # Si solo hay dos clases, se devuelve una matriz con una columna
    targets = reshape(targets .== classes[1], :, 1)
else
    # Si hay mas de dos clases se devuelve una matriz con una columna por clase
    # Cualquiera de estos dos tipos (Array{Bool,2} o BitArray{2}) vale perfectamente
    oneHot = Array{Bool,2}(undef, length(targets), numClasses)
    oneHot = BitArray{2}(undef, length(targets), numClasses)
    for numClass = 1:numClasses
        oneHot[:, numClass] .= (targets .== classes[numClass])
    end
    targets = oneHot
end;
println("Tamaño de la matriz de salidas deseadas despues de codificar: ", size(targets, 1), "x", size(targets, 2), " de tipo ", typeof(targets));

# Comprobamos que ambas matrices tienen el mismo número de filas
@assert (size(inputs, 1) == size(targets, 1)) "Las matrices de entradas y salidas deseadas no tienen el mismo numero de filas"

# Calculamos los valores de normalizacion, de uno u otro tipo
minValues = minimum(inputs, dims=1);
maxValues = maximum(inputs, dims=1);
avgValues = mean(inputs, dims=1);
stdValues = std(inputs, dims=1);

# Y normalizamos de un tipo u otro. Por ejemplo, mediante maximo y minimo:
inputs .-= minValues;
inputs ./= (maxValues .- minValues);
# Si hay algun atributo en el que todos los valores son iguales, se pone a 0
inputs[:, vec(minValues .== maxValues)] .= 0;
# Dejamos aqui indicado como se haria para normalizar mediante media y desviacion tipica
# inputs .-= avgValues;
# inputs ./= stdValues;
# # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
# inputs[:, vec(stdValues.==0)] .= 0;
