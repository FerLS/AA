using XLSX
using DelimitedFiles
using FileIO
using Flux
using Flux.Losses


ann = Chain();
ann = Chain(ann..., Dense(4, 5, σ));
ann = Chain(ann..., Dense(5, 3, identity));
ann = Chain(ann..., softmax);


outputs = ann(inputs');




function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    if length(classes) == 2
        # Compare feature with one of the classes to generate a boolean vector
        bool_vec = feature .== classes[1]
        # Transform the vector into a 2D array with one column and return it
        return reshape(bool_vec, :, 1)
    elseif length(classes) > 2
        # Create a boolean matrix with as many rows as patterns and as many columns as classes
        bool_mat = convert(Array{Bool,2}, zeros(length(feature), length(classes)))
        # Iterate over each column/class
        for (i, class) in enumerate(classes)
            # Assign the values of that column as the result of comparing the feature vector with the corresponding class
            bool_mat[:, i] = feature .== class
        end
        return bool_mat
    end
end


#



oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)

#test 


function calculateMinMaxNormalizationParameters(data::AbstractArray{<:Real,2})
    mins = minimum(data, dims=1)
    maxs = maximum(data, dims=1)
    return (mins, maxs)
end




function calculateZeroMeanNormalizationParameters(data::AbstractArray{<:Real,2})
    means = mean(data, dims=1)
    stds = std(data, dims=1)
    return (means, stds)
end




function normalizeMinMax!(data::AbstractArray{<:Real,2}, params::NTuple{2,AbstractArray{<:Real,2}})
    mins, maxs = params
    data .= (data .- mins) ./ (maxs .- mins)
    return data
end

function normalizeMinMax!(data::AbstractArray{<:Real,2})
    params = calculateMinMaxNormalizationParameters(data)
    normalizeMinMax!(data, params)
end

function normalizeMinMax(data::AbstractArray{<:Real,2}, params::NTuple{2,AbstractArray{<:Real,2}})
    new_data = copy(data)
    normalizeMinMax!(new_data, params)
    return new_data
end

function normalizeMinMax(data::AbstractArray{<:Real,2})
    params = calculateMinMaxNormalizationParameters(data)
    normalizeMinMax(data, params)
end


function classifyOutputs(outputs::AbstractArray{<:Real,1}, threshold::Real=0.5)
    return outputs .>= threshold
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}, threshold::Real=0.5)
    if size(outputs, 2) == 1
        vector_outputs = classifyOutputs(outputs[:], threshold)
        return reshape(vector_outputs, :, 1)
    else
        max_indices = argmax(outputs, dims=2)
        bool_matrix = fill(false, size(outputs))
        bool_matrix[CartesianIndex.(eachindex(max_indices), max_indices)] .= true
        return bool_matrix
    end
end


function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{Bool,1})
    return mean(targets .== outputs)
end

function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{Bool,2})
    if size(targets, 2) == 1
        return accuracy(targets[:, 1], outputs[:, 1])
    elseif size(targets, 2) > 2
        return mean(all(targets .== outputs, dims=2))
    end
end

function accuracy(targets::AbstractArray{Bool,1}, outputs::AbstractArray{<:Real,1}, threshold::Real=0.5)
    binary_outputs = classifyOutputs(outputs, threshold)
    return accuracy(targets, binary_outputs)
end

function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{<:Real,2})
    if size(outputs, 2) == 1
        return accuracy(targets[:, 1], outputs[:, 1])
    else
        binary_outputs = classifyOutputs(outputs)
        return accuracy(targets, binary_outputs)
    end
end

(_, indicesMaxEachInstance) = findmax(outputs, dims=2);
outputs = falses(size(outputs));
outputs[indicesMaxEachInstance] .= true;
classComparison = targets' .== outputs


meanAccuracy = mean(correctClassifications)
classComparison = targets' .!= outputs



function buildClassANN(topology::AbstractArray{<:Int,1}, numInputs::Int, numOutputs::Int, σ = relu)
    ann = Chain()
    numInputsLayer = numInputs

    if !isempty(topology)
        for numOutputsLayer in topology
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ))
            numInputsLayer = numOutputsLayer
        end
    end

    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, softmax))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs))
    end

    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                      maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
    inputs, targets = dataset
    numInputs = size(inputs, 2)
    numOutputs = size(targets, 2)
    
    ann = buildClassANN(topology, numInputs, numOutputs)
    loss_history = []
    
    for epoch in 1:maxEpochs
        Flux.train!(loss, Flux.params(ann), [(inputs', targets')])
        loss_value = loss(ann(inputs'), targets')
        push!(loss_history, loss_value)
        
        if loss_value <= minLoss
            break
        end
    end
    
    return (ann, loss_history)
end