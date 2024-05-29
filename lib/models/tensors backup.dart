// Importa o pacote image
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import 'models.dart';

enum RandomDistribution {
  uniform,
  normal,
  bernoulli,
  poisson,
  exponential,
}

enum ScalarOperation {
  multiplication,
  division,
  addition,
  subtraction,
}

class SusiImageInfo {
  String type;
  int width;
  int height;
  int colorDepth;

  SusiImageInfo(this.type, this.width, this.height, this.colorDepth);
}

typedef CustomFunction = double Function(double value);

class ConcreteTensor extends Tensor {
  ConcreteTensor(List<double> data, List<int>? shape) : super(data, shape);

  List<double> elements() {
    return _data;
  }

  List<int> getShape() {
    return _shape;
  }

  List<double> row(int rowIndex) {
    if (rowIndex < 0 || rowIndex >= _shape[0]) {
      throw RangeError('Row index out of range');
    }
    List<double> result = List<double>.filled(_shape[1], 0);
    int startIndex = rowIndex * _shape[1];
    int endIndex = startIndex + _shape[1];
    result.setRange(0, _shape[1], _data.sublist(startIndex, endIndex));
    return result;
  }

  @override
  Map<String, dynamic> toJSON() {
    return {
      'data': _data,
      'shape': _shape,
    };
  }

  @override
  void fromJSON(Map<String, dynamic> json) {
    _data.clear();
    _data.addAll(List<double>.from(json['data']));
    _shape.clear();
    _shape.addAll(List<int>.from(json['shape']));
  }
}

abstract class Tensor {
  final List<double> _data;

  final List<int> _shape;

  SusiImageInfo? imageInfo;

  Tensor(this._data, [List<int>? shape, this.imageInfo])
      : _shape = shape ?? [_data.length];

  List<double> get data => _data;
  List<int> get shape => _shape;

  int get size => _data.length;

  double getValue(int index) {
    if (index < 0 || index >= size) {
      throw RangeError('Index out of range');
    }
    return _data[index];
  }

  void setValue(int index, double value) {
    if (index < 0 || index >= size) {
      throw RangeError('Index out of range');
    }
    _data[index] = value;
  }

  Tensor operator +(Tensor other) {
// Get the size of the smaller tensor
    int smallerSize = math.min(size, other.size);

// Create an empty tensor with the size of the smaller tensor
    Tensor result = Tensor.zeros([smallerSize]);

// Perform element-wise addition
    for (int i = 0; i < smallerSize; i++) {
      result._data[i] = _data[i] + other._data[i];
    }

    return result;
  }

  Tensor operator -(Tensor other) {
// Get the size of the smaller tensor
    int smallerSize = math.min(size, other.size);

// Create an empty tensor with the size of the smaller tensor
    Tensor result = Tensor.zeros([smallerSize]);

// Perform element-wise subtraction
    for (int i = 0; i < smallerSize; i++) {
      result._data[i] = _data[i] - other._data[i];
    }

    return result;
  }

  Tensor operator *(Tensor other) {
// Get the size of the smaller tensor
    int smallerSize = math.min(size, other.size);

// Create an empty tensor with the size of the smaller tensor
    Tensor result = Tensor.zeros([smallerSize]);

// Perform element-wise multiplication
    for (int i = 0; i < smallerSize; i++) {
      result._data[i] = _data[i] * other._data[i];
    }

    return result;
  }

  Tensor operator /(Tensor other) {
// Get the size of the smaller tensor
    int smallerSize = math.min(size, other.size);

// Create an empty tensor with the size of the smaller tensor
    Tensor result = Tensor.zeros([smallerSize]);

// Perform element-wise division
    for (int i = 0; i < smallerSize; i++) {
      result._data[i] = _data[i] / other._data[i];
    }

    return result;
  }

  static Tensor createTensor(List<double> data, {required List<int> shape}) {
// print('createTensor shape $shape');
// shape ??= List.generate(data.length, (i) => i);
    return ConcreteTensor(data, shape);
  }

  factory Tensor.fromList(List<double> list, {List<int>? shape}) {
    return ConcreteTensor(list, shape);
  }

  static Tensor fromImage(
    Uint8List image, {
    required int height,
    required int width,
    required int colors,
  }) {
    List<double> imageList = Tensor.convertUint8ListToDoubleList(image);

    return Tensor.fromList(imageList, shape: [height, width, colors]);
  }

  /// Converte as labels para o formato "one-hot"
  Tensor convertLabelsToOneHot(int numClasses) {
    List<double> result = List<double>.filled(size * numClasses, 0);

    for (int i = 0; i < size; i++) {
      int label = _data[i].toInt();
      int index = i * numClasses + label;
      result[index] = 1;
    }

    return Tensor.createTensor(result, shape: [size, numClasses]);
  }

  Tensor bilinearInterpolation(List<int> newShape) {
// Validate the new shape
    for (int i = 0; i < newShape.length; i++) {
      if (newShape[i] <= 0) {
        throw ArgumentError(
            'The new shape must be a list of positive integers.');
      }
    }

// Calculate the number of elements in the resized tensor
    int newElements = 1;
    for (int i = 0; i < newShape.length; i++) {
      newElements *= newShape[i];
    }

// Create the resized tensor
    Tensor resizedTensor =
        ConcreteTensor(List<double>.filled(newElements, 0), newShape);

// Iterate over the resized tensor
    for (int i = 0; i < newElements; i++) {
// Calculate the coordinates of the interpolation point in the original tensor
      double newX = i % newShape[0] / newShape[0];
      double newY = i / newShape[0];

// Calculate the coordinates of the interpolation point in the original tensor
      int x = newX.floor();
      int y = newY.floor();

// Calculate the values of the original pixels at the interpolation points
      double pixel00 = getValue(x * _shape[1] + y);
      double pixel10 = getValue((x + 1) * _shape[1] + y);
      double pixel01 = getValue(x * _shape[1] + y + 1);
      double pixel11 = getValue((x + 1) * _shape[1] + y + 1);

// Calculate the value of the resized pixel
      double resizedPixel = (pixel00 * (1 - newX) * (1 - newY) +
          pixel10 * newX * (1 - newY) +
          pixel01 * (1 - newX) * newY +
          pixel11 * newX * newY);

// Set the value of the resized pixel
      resizedTensor.setValue(i, resizedPixel);
    }

    return resizedTensor;
  }

  /// Redimensiona o tensor usando a estratégia de "resize"
  Tensor resize(int newSize, {bool removeLasts = true, bool addLasts = true}) {
    List<int> currentShape = _shape;

    print('Current Shape: $currentShape');
    print('New Shape: $newSize');

    List<double> newData = List<double>.filled(
      _calculateSize([newSize]),
      0,
    );

// Resize the data based on the provided parameters
    if (newSize > currentShape.first) {
      if (addLasts) {
        for (int i = 0; i < _data.length; i++) {
          newData[i] = newData[i] + _data[i];
        }
      } else {
        int startIndex = newSize - _data.length;

        for (int i = startIndex; i < newSize; i++) {
          int currentElementIndex = i - startIndex;
          newData[i] = newData[i] + _data[currentElementIndex];
        }
      }
    } else {
      if (removeLasts) {
        print('Removing Lasts...');
        int endIndex = newSize;
        for (int i = 0; i < endIndex; i++) {
          newData[i] = _data[i];
        }
      } else {
        print('Removing Firsts...');
        int indexStart = 0;
        if (newSize < currentShape.first) {
          indexStart = currentShape.first - newSize;
        }
        for (int i = indexStart; i < currentShape.first; i++) {
          int currentElementIndex = i;
          int newElementIndex = i - indexStart;
          newData[newElementIndex] = _data[currentElementIndex];
        }
      }
    }

// Update the shape to reflect the new size

    print('New Data: $newData');
    return Tensor.createTensor(newData, shape: [newSize]);
  }

  static Tensor range(double start, double stop, {double step = 1.0}) {
    List<double> data = <double>[];
    for (double i = start; i < stop; i += step) {
      data.add(i);
    }
    return Tensor.fromList(data);
  }

  static Tensor zeros(List<int> shape) {
// print('zeros shape $shape');

    int size = _calculateSize(shape);
    List<double> data = List<double>.filled(size, 0);
    Tensor zeros = createTensor(data, shape: shape);
// print('zeros shape ${zeros.shape}');
    return zeros;
  }

  static Tensor constant(List<int> shape, double value) {
    int size = _calculateSize(shape);
    List<double> data = List<double>.filled(size, value);
    return Tensor.createTensor(data, shape: shape);
  }

  static int _calculateSize(List<int> shape) {
    int size = 1;
    for (int dim in shape) {
      size *= dim;
    }
    return size;
  }

  List<int> getPatchIndices(List<int> kernelSize, int start0, int end0) {
    List<int> indices = [];
    for (int i = start0; i < end0; i += kernelSize[0]) {
      indices.add(i);
    }
    return indices;
  }

  List<Tensor> patches(
    List<int> kernelSize,
    List<int> strides,
    List<int> indices,
  ) {
    List<Tensor> patches = [];

    for (int rowIndex in indices) {
      // Tensor row = getRows([rowIndex]);

      int start0 = rowIndex * kernelSize[0];
      int end0 = start0 + kernelSize[0];
      int start1 = rowIndex * (kernelSize[1] - 1) ~/ 2;
      int end1 = start1 + kernelSize[1];

      if (end0 > _shape[0] * _shape[1] * _shape[2]) {
        break;
      }
      if (end1 > _shape[0] * _shape[1] * _shape[2]) {
        break;
      }

      // Tensor patch = row.slice(start0, end0, strides, start1, end1);

      // patches.add(row);
    }

    return patches;
  }

  Tensor getRows(List<int> indices) {
    Tensor newTensor = Tensor.zeros([_shape[0], _shape[1], _shape[2]]);
    for (int index in indices) {
      if (index >= 0 && index < _shape[0] * _shape[1] * _shape[2]) {
        for (int i = 0; i < newTensor._data.length; i++) {
          if (i > _data.length - 1) {
            break;
          }
          newTensor._data[i] = _data[i];
        }
      }
    }
    return newTensor;
  }

  Tensor slice(int start0, int end0, List<int> strides, int start1, int end1) {
    if (start0 < 0) {
      throw Exception("start0 must be greater than or equal to 0");
    }

    if (end0 > _data.length) {
      throw Exception(
          "end0 $end0 must be less than or equal to the size of dimension 0 of the input ${_data.length}");
    }

    if (end1 > _data.length) {
      throw Exception(
          "end1 must be less than or equal to the size of dimension 1 of the input");
    }

    int paddingSize = 0;

    int subtensorStart0 = (start0 - paddingSize / 2).toInt() + paddingSize;
    int subtensorEnd0 = (end0 + paddingSize / 2).toInt() + paddingSize;
    int subtensorStart1 = (start1 - paddingSize / 2).toInt();
    int subtensorEnd1 = (end1 + paddingSize / 2).toInt();

    int subtensorLength0 = subtensorEnd0 - subtensorStart0;
    int subtensorLength1 = subtensorEnd1 - subtensorStart1;

    // Creating the subtensor with calculated dimensions
    Tensor subtensor = Tensor.zeros([subtensorLength0, subtensorLength1]);

    // Copying values from the original tensor to the subtensor

    for (int j = 0; j < subtensorEnd1 - subtensorStart1; j++) {
      if (j + subtensorStart0 >= _data.length) {
        break;
      }

      subtensor.setValue(j, _data[j + subtensorStart0]);
    }

    return subtensor;
  }

//   Tensor slice(int start0, int end0, List<int> strides, int start1, int end1) {
//     if (start0 < 0) {
//       throw Exception("start0 must be greater than or equal to 0");
//     }

//     if (end0 > _data.length) {
//       throw Exception(
//           "end0 $end0 must be less than or equal to the size of dimension 0 of the input ${_data.length}");
//     }

//     if (end1 > _data.length) {
//       throw Exception(
//           "end1 must be less than or equal to the size of dimension 1 of the input");
//     }

//     int paddingSize = 0;

//     int subtensorStart0 =
//         math.max((start0 - paddingSize / 2).toInt() + paddingSize, 0);

//     int subtensorEnd0 =
//         math.min((end0 + paddingSize / 2).toInt() + paddingSize, _data.length);

//     int subtensorStart1 = math.max((start1 - paddingSize / 2).toInt(), 0);

//     int subtensorEnd1 =
//         math.min((end1 + paddingSize / 2).toInt(), _shape[0] * _shape[1]);

//     Tensor subtensor = Tensor.zeros([
//       math.max(subtensorEnd0 - subtensorStart0, 0),
//       2, // math.max(subtensorEnd1 - subtensorStart1, 0),
// // strides[1]
//     ]);

//     if (subtensorStart0 < 0) {
//       paddingSize = -subtensorStart0;
//     }

//     if (subtensorEnd0 > _data.length) {
//       paddingSize += subtensorEnd0 - _data.length;
//     }
//     if (subtensorStart1 < 0) {
//       paddingSize = -subtensorStart1;
//     }

//     if (subtensorEnd1 > _data.length) {
//       paddingSize += subtensorEnd1 - _data.length;
//     }
//     if (_data.length < paddingSize) {
//       _data.addAll(
//           List<double>.filled(math.max(paddingSize - _data.length, 0), 0.0));
//     }

//     for (int i = 0; i < subtensorEnd1 - subtensorStart1; i++) {
//       if (i > shape[0]) {
//         break;
//       }
//       subtensor.setValue(i, _data[subtensorStart1 + i]);
//     }

//     for (int j = subtensorEnd0 - subtensorStart0;
//         j < subtensorEnd1 - subtensorStart1 + subtensorEnd0 - subtensorStart0;
//         j++) {
//       if (j > shape[1]) {
//         break;
//       }

//       subtensor.setValue(j, _data[subtensorStart0 + j]);
//     }
//     return subtensor;
//   }

// Tensor slice(int start0, int end0, List<int> strides,
//     [int start1 = 0, int? end1]) {
//   print('start slice');

//   // Validate the indices
//   if (start0 < 0) {
//     throw Exception("start0 must be greater than or equal to 0");
//   }

//   if (end0 > _data.length) {
//     throw Exception(
//         "end0 $end0 must be less than or equal to the size of dimension 0 of the input ${_data.length}");
//   }

//   // Adjust the end1 index if it's not specified
//   end1 ??= _shape[1];

//   // Validate the end1 index
//   if (end1 > _data.length) {
//     throw Exception(
//         "end1 must be less than or equal to the size of dimension 1 of the input");
//   }

//   // Calculate the unpadded start and end indices
//   int unpaddedStart0 = math.max(start0, 0);
//   int unpaddedEnd0 = math.min(end0, _data.length);

//   // Calculate the subtensor start and end indices without padding
//   int subtensorStart0 = unpaddedStart0;
//   int subtensorEnd0 = unpaddedEnd0;
//   int subtensorStart1 = start1;
//   int subtensorEnd1 = end1;

//   // Create an empty tensor to store the subtensor
//   Tensor subtensor = Tensor.zeros(
//       [subtensorEnd0 - subtensorStart0, subtensorEnd1 - subtensorStart1]);

//   // Check if the subtensor start index is within bounds
//   if (subtensorStart1 >= subtensorEnd1) {
//     return subtensor;
//   }

//   // Copy the data from the original tensor to the subtensor without padding
//   for (int i = subtensorStart0; i < subtensorEnd0; i++) {
//     for (int j = subtensorStart1;
//         j < math.min(subtensorEnd1, _data.length);
//         j++) {
//       int subtensorIndex =
//           (i - subtensorStart0) * strides[0] + (j - subtensorStart1);
//       subtensor.setValue(subtensorIndex, _data[i * strides[0] + j]);
//     }
//   }

//   print('finished slice');

//   return subtensor;
// }

  Future<Tensor> convolve2d(
      List<int> kernelSize, int filters, ActivationFunction activation,
      [Tensor? weights, Tensor? biases]) async {
    weights ??= random([kernelSize[0], kernelSize[1], shape[-1], filters]);
    biases ??= Tensor.zeros([filters]);

    print('input shape $shape, size $size');
    print('starting convolve2d');
// Calculate the start and end indices of the patches
    int start0 = (kernelSize[0] - 1) ~/ 2;
    int end0 = (shape[0] * shape[1] * shape[2]) - (kernelSize[0] - 1);

    print('getPatcheIndices');
// Get the indices of the patches
    List<int> patchIndices = getPatchIndices(kernelSize, start0, end0);

// print('patchIndices $patchIndices');
    print('calculating strides');
// Get the strides
    List<int> strides = _calculateStrides(shape);
    print('$strides');

    print('calculating patches');
// Split the input into patches

    // List<Tensor> patches = this.patches(kernelSize, strides, patchIndices);

// Tensor patches = this.patches(kernelSize, strides, patchIndices).first;
// List<Tensor> patches = this.patches(kernelSize, strides, patchIndices);
// print('patchs lenght ${patches.data}');
    print('multiplying patches');
// Multiply each patch by the weights of the layer
    List<Tensor> multipliedPatches = [];
// print('patches ${patches.length}');
// print('patches ${patches[0].size}');
// print('patches ${patches[0].data}');

//     for (Tensor patch in patches) {
// // print('patch ${patch._data}');
// // print('patch shape ${patch.shape}');
// // print('patchIndices $patchIndices');
// // print('patch info ${patches.indexOf(patch)}');

// // print('weights ${weights._data}');

//       patch._data.fold(0, (previousValue, element) {
// // print("previousValue $previousValue element $element");

//         //!  // element = element * weights!.data[previousValue];

// // print("previousValue $previousValue element $element");
//         previousValue += 1;
//         return previousValue;
//       });
// //elementWiseMultiply(patch, weights);

//       multipliedPatches.add(patch);
//     }

//     Tensor output = Tensor.zeros(
//         [multipliedPatches.first.shape[0], multipliedPatches.length]);

// // print('output ${output._shape}');
// // print('multiplied patches ${multipliedPatches.last.data}');
//     print('multiplied patches ${multipliedPatches.last.shape}');
//     print('multiplied patches ${multipliedPatches.length}');
// // print('output zeros ${output._shape}');
// // print('output zeros ${output._data}');

//     for (int i = 0; i < multipliedPatches.length - 1; i++) {
//       for (int j = 0; j < multipliedPatches[i].size; j++) {
//         int index = i * multipliedPatches[i].size + j;
//         if (index >= output.size) {
//           break;
//         }
// // print('multipliedPatches[i] ${multipliedPatches[i]._data}');
// // print(
// //     'i $i j $j i * multipliedPatches[i].size +j ${i * multipliedPatches[i].size + j} multipliedPatches[$i]._data[$j] ${multipliedPatches[i]._data[j]} multipliedPatches[$i].size ${multipliedPatches[i].size} multipliedPatches.length ${multipliedPatches.length} multipliedPatches[$i]._data.lenght ${multipliedPatches[i]._data.length}');

//         output.setValue(index, multipliedPatches[i]._data[j]);
//       }
//     }

// // Sum the products of the patches by the weights to obtain a value for each filter
// // Tensor filteredValues = sum(multipliedPatches);
// // print(
// //     'summed ${output._data} output.shape ${output.shape} output.size ${output.size}');
// // print('filteredValues ${filteredValues._data}');

// // Add the biases to the filter values to obtain the output of the layer

//     output = output + biases;
//     print('applying activation');
// // print('output from biases ${output._data}');
// // Apply the activation function to the output of the layer

//     switch (activation) {
//       case ActivationFunction.relu:
// // print('output antes do relu ${output.data}');
//         //! // output = output.relu();
// // print('output depois do relu ${output.data}');
//         break;
//       case ActivationFunction.tanh:
//         output = output.tanh();
//         break;
//       case ActivationFunction.sigmoid:
//         output = output.sigmoid();
//         break;
//       default:
//         break;
//     }

    Tensor output = this;
// print('output convolve2d finished ${output.data}');

    return output;
  }

  Tensor maxPooling2d(Tensor input, List<int> poolSize) {
// Verificar se o input tem as dimensões corretas
    List<int> inputShape = input.shape;
    if (inputShape[0] % poolSize[0] != 0) {
      throw Exception(
          "O input da camada MaxPooling2D deve ser divisível pelo poolSize");
    }
    if (inputShape.length < 2) {
      throw Exception(
          "The input to the MaxPooling2D layer must have at least two dimensions");
    }

// Calcular o número de elementos na saída
    int outputLength =
        ((inputShape[0] - poolSize[0]) / poolSize[0] + 1).toInt();
    List<int> strides = _calculateStrides(shape);

// Criar um tensor de saída com as dimensões corretas
    Tensor output = Tensor.zeros([outputLength, outputLength]);

// Iterar sobre as janelas de entrada
    for (int i = 0; i < output.shape[0] * output.shape[1]; i++) {
// Get the start and end indices of the window
      int start0 = i * poolSize[0];
      int end0 = start0 + poolSize[0];
      int start1 = i * (poolSize[1] - 1) ~/ 2;
      int end1 = start1 + poolSize[1];

// Get the window slice
      List<double> window =
          input.slice(start0, end0, strides, start1, end1)._data;

// Set the output value
      output.setValue(i, window.fold(0, (a, b) => math.max(a, b)));
// print('i $i window.lenght ${window.length}');
    }

    return output;
  }
// Tensor maxPooling2d(Tensor input, List<int> poolSize) {
//   // Verificar se o input tem as dimensões corretas
//   List<int> inputShape = input.shape;
//   if (inputShape[0] % poolSize[0] != 0) {
//     throw Exception(
//         "O input da camada MaxPooling2D deve ser divisível pelo poolSize");
//   }
//   if (inputShape.length < 2) {
//     throw Exception(
//         "The input to the MaxPooling2D layer must have at least two dimensions");
//   }

//   // Calcular o número de elementos na saída
//   int outputLength =
//       ((inputShape[0] - poolSize[0]) / poolSize[0] + 1).toInt();
//   List<int> strides = _calculateStrides(shape);

//   // Criar um tensor de saída com as dimensões corretas
//   Tensor output = Tensor.zeros([outputLength, outputLength]);

//   // Iterar sobre as janelas de entrada
//   for (int i = 0; i < output._shape[0]; i++) {
//     // Obter o índice inicial da janela
//     int start0 = i * strides[0];
//     int start1 = (i * strides[1] + (poolSize[1] - 1) ~/ 2) % inputShape[1];

//     // Verificar se a entrada é maior que o pool
//     if (start1 + poolSize[1] > inputShape[1]) {
//       start1 = inputShape[1] - poolSize[1];
//     }

//     // Obter a janela de entrada
//     List<double> window = input
//         .slice(
//           start0,
//           poolSize[0],
//           strides,
//         )
//         ._data;

//     // Obter o valor máximo da janela
//     print('window $window');
//     print('outputLenght $outputLength i $i');
//     // Atribuir o valor máximo ao tensor de saída
//     output._data[i] = window.fold(0, (a, b) => math.max(a, b));

//     print('outputLenght $outputLength i $i');
//   }

//   return output;
// }

// Tensor maxPooling2d(Tensor input, List<int> poolSize) {
//   // Verificar se o input tem as dimensões corretas
//   List<int> inputShape = input.shape;
//   if (inputShape[0] % poolSize[0] != 0) {
//     throw Exception(
//         "O input da camada MaxPooling2D deve ser divisível pelo poolSize");
//   }
//   if (inputShape.length < 2) {
//     throw Exception(
//         "The input to the MaxPooling2D layer must have at least two dimensions");
//   }

//   // Calcular o número de elementos na saída
//   int outputLength =
//       ((inputShape[0] - poolSize[0]) / poolSize[0] + 1).toInt();

//   // Criar um tensor de saída com as dimensões corretas
//   Tensor output = Tensor.zeros([outputLength]);

//   // Iterar sobre as janelas de entrada
//   for (int i = 0; i < outputLength; i++) {
//     // Obter a janela de entrada
//     List<double> window =
//         input.slice(i * poolSize[0], poolSize[0], [0, inputShape[1]])._data;

//     // Obter o valor máximo da janela
//     double maxValue = window.fold(0, (a, b) => math.max(a, b));

//     // Atribuir o valor máximo ao tensor de saída
//     output._data[i] = maxValue;
//   }

//   return output;
// }

// Tensor maxPooling2d(Tensor input, List<int> poolSize) {
//   // Verify if the input has the correct dimensions
//   List<int> inputShape = input.shape;
//   if (inputShape.length < 2) {
//     throw Exception(
//         "The input to the MaxPooling2D layer must have at least two dimensions");
//   }

//   print("input shape ${input.shape}");

//   // Calculate the number of elements in the output
//   int outputLength =
//       ((inputShape[0] - poolSize[0]) / poolSize[0] + 1).toInt();

//   // Calculate the output shape
//   List<int> outputShape = inputShape.sublist(0, inputShape.length - 1);
//   outputShape.add(outputLength);

//   // Create an output tensor with the correct dimensions
//   Tensor output = Tensor.zeros(outputShape);

//   // Calculate strides
//   List<int> strides = _calculateStrides(shape);

//   // Iterate over the input
//   for (int i = 0; i < inputShape[0]; i += poolSize[0]) {
//     // Get the input window
//     List<double> window =
//         input.slice(i, poolSize[0], strides, 0, poolSize[1])._data;

//     // Get the maximum value of the window
//     double maxValue = window.fold(0, (a, b) => math.max(a, b));

//     // Assign the maximum value to the output tensor
//     output._data[(i ~/ poolSize[0]) * outputShape[1]] = maxValue;
//   }

//   return output;
// }

// Tensor maxPooling2d(Tensor input, List<int> poolSize) {
//   // Verify if the input has the correct dimensions
//   List<int> inputShape = input.shape;
//   if (inputShape.length < 2) {
//     throw Exception(
//         "The input to the MaxPooling2D layer must have at least two dimensions");
//   }

//   // Calculate the output shape
//   List<int> outputShape = inputShape;

//   // Create an output tensor with the correct dimensions
//   Tensor output = Tensor.zeros(outputShape);

//   // Calculate strides
//   List<int> strides = _calculateStrides(shape);

//   for (int i = 0; i < outputShape[0]; i++) {
//     // Get the input window
//     int start0 = i * poolSize[0];
//     int start1 = (poolSize[1] - 1) ~/ 2;

//     // Check if the input window is out of bounds
//     if (start0 + poolSize[0] > inputShape[0]) {
//       poolSize[0] = inputShape[0] - start0;
//       print('erro');
//     }

//     print('i $i');
//     // Get the window slice
//     Tensor window = input.slice(
//       start0,
//       poolSize[0],
//       strides,
//       start1,
//       poolSize[1],
//     );
//     print('i $i');
//     // Set the output value
//     output._data[i] = window._data.fold(0, (a, b) => math.max(a, b));
//     print('i $i');
//   }

//   return output;
// }

  Tensor sum(List<Tensor> tensors) {
// Get the size of the smallest tensor
    int smallerSize = math.min(tensors[0].size, tensors.last.size);

// Create an empty tensor with the size of the smaller tensor
    Tensor sumTensor = Tensor.zeros([smallerSize]);

// Iterate over the elements of the tensors
    for (int i = 0; i < smallerSize; i++) {
// Add the corresponding elements of the tensors
      sumTensor._data[i] = tensors[0]._data[i] + tensors.last._data[i];
    }

    return sumTensor;
  }

  Tensor pow(double exponent) {
    List<double> result = List<double>.filled(size, 0);

    for (int i = 0; i < size; i++) {
      result[i] = math.pow(_data[i], exponent).toDouble();
    }

    return Tensor.createTensor(result, shape: _shape);
  }

  Tensor random(List<int> shape,
      {RandomDistribution distribution = RandomDistribution.uniform}) {
    int size = _calculateSize(shape);
    List<double> data =
        List<double>.generate(size, (_) => _generateRandomValue(distribution));
    return createTensor(data, shape: shape);
  }

  Tensor transpose() {
    List<double> data = List<double>.from(this.data);
    List<int> shape = List<int>.from(this.shape.reversed);
    return createTensor(data, shape: shape);
  }

  double mean() {
    double sum = data.reduce((value, element) => value + element);
    return sum / size;
  }

  double median() {
    List<double> sortedData = List<double>.from(data);
    sortedData.sort();

    if (size % 2 == 0) {
      int midIndex = size ~/ 2;
      return (sortedData[midIndex - 1] + sortedData[midIndex]) / 2;
    } else {
      int midIndex = size ~/ 2;
      return sortedData[midIndex];
    }
  }

  double standardDeviation() {
    double average = mean();
    double sumOfSquaredDifferences = 0;

    for (double value in data) {
      double difference = value - average;
      sumOfSquaredDifferences += difference * difference;
    }

    double variance = sumOfSquaredDifferences / size;
    return math.sqrt(variance);
  }

  double _generateRandomValue(RandomDistribution distribution) {
    final random = math.Random();
    switch (distribution) {
      case RandomDistribution.uniform:
        return random.nextDouble();
      case RandomDistribution.normal:
        double newRandom = random.nextDouble();
        return math.sqrt(-2 * math.log(newRandom)) *
            math.cos(2 * math.pi * newRandom);
      case RandomDistribution.bernoulli:
        return random.nextDouble() < 0.5 ? 0.0 : 1.0;
      case RandomDistribution.poisson:
        double lambda = 1.0;
        double L = math.exp(-lambda);
        double p = 1.0;
        int k = 0;
        do {
          k++;
          p *= random.nextDouble();
        } while (p > L);
        return (k - 1).toDouble();
      case RandomDistribution.exponential:
        double lambda = 1.0;
        return -math.log(random.nextDouble()) / lambda;
      default:
        return random.nextDouble();
    }
  }

// Operações matemáticas
  Tensor applyScalarOperation(
      {required double scalar,
      ScalarOperation operation = ScalarOperation.multiplication,
      CustomFunction? customFunction}) {
    for (int i = 0; i < size; i++) {
      switch (operation) {
        case ScalarOperation.multiplication:
          _data[i] *= scalar;
          break;
        case ScalarOperation.division:
          _data[i] /= scalar;
          break;
        case ScalarOperation.addition:
          _data[i] += scalar;
          break;
        case ScalarOperation.subtraction:
          _data[i] -= scalar;
          break;
      }
      if (customFunction != null) {
        _data[i] = customFunction(_data[i]);
      }
    }
    return this;
  }

  Tensor normalize() {
    double minValue = _data.reduce(math.min);
    double maxValue = _data.reduce(math.max);
    double range = maxValue - minValue;
    for (int i = 0; i < size; i++) {
      _data[i] = (_data[i] - minValue) / range;
    }
    return this;
  }

  Tensor binarize() {
    for (int i = 0; i < size; i++) {
      _data[i] = _data[i] >= 0.5 ? 1.0 : 0.0;
    }
    return this;
  }

  /// * **How to use Resize function**
  /// Resize the tensor by adding or removing elements.
  ///
  /// [valueToAdd]: The value to be added. Defaults to null, which adds zeros.
  /// [countToAdd]: The number of elements to add. Defaults to 0.
  /// [countToRemove]: The number of elements to remove. Defaults to 0.
  /// [addToBeginning]: Whether to add elements to the beginning of the tensor. Defaults to false.
  /// [removeFromBeginning]: Whether to remove elements from the beginning of the tensor. Defaults to false.
  ///
  /// Examples:
  ///
  /// Add 2 doubles as 1.0 at the end of the tensor
  ///
  /// ```dart
  /// print(tensor.data); // [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
  /// tensor.resize( 1.0, countToAdd: 2, addToBeginning: false);
  /// print(tensor.data); // [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
  /// ```
  ///
  /// Remove 2 elements at the beginning of the tensor
  ///
  /// ```dart
  /// print(tensor.data); // [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
  /// tensor.resize(countToRemove: 2, removeToBeginning: true);
  /// print(tensor.data); // [0.0, 0.0, 0.0, 0.0]
  /// ```
  ///
  Tensor resizeCustom({
    double? valueToAdd,
    int countToAdd = 0,
    int countToRemove = 0,
    bool addToBeginning = false,
    bool removeFromBeginning = false,
  }) {
    if (countToRemove > 0) {
// Remove elementos do início do tensor
      if (removeFromBeginning) {
        _data.removeRange(0, countToRemove);
      }
// Remove elementos do final do tensor
      else {
        final startIndex = size - countToRemove;
        _data.removeRange(startIndex, size);
      }
    }

    if (countToAdd > 0) {
// Adiciona elementos no início do tensor
      if (addToBeginning) {
        _data.insertAll(0, List<double>.filled(countToAdd, valueToAdd ?? 0.0));
      }
// Adiciona elementos no final do tensor
      else {
        _data.addAll(List<double>.filled(countToAdd, valueToAdd ?? 0.0));
      }
    }

    return this;
  }

  Tensor customMultiplyTensor({required CustomFunction customFunction}) {
    for (int i = 0; i < size; i++) {
      _data[i] = customFunction(_data[i]);
    }
    return this;
  }

  double normL2() {
    double sumOfSquares = 0;
    for (int i = 0; i < size; i++) {
      sumOfSquares += _data[i] * _data[i];
    }
    return math.sqrt(sumOfSquares);
  }

  double normInfinite() {
    double maxAbsValue = 0;
    for (int i = 0; i < size; i++) {
      double absValue = _data[i].abs();
      if (absValue > maxAbsValue) {
        maxAbsValue = absValue;
      }
    }
    return maxAbsValue;
  }

  bool canBroadcast(List<int> a, List<int> b) {
// Get the lengths of the two shapes
    int aLength = a.length;
    int bLength = b.length;

// If the lengths of the two shapes are not equal, broadcasting is not possible
    if (aLength != bLength) {
      return false;
    }

// Iterate over the dimensions of the two shapes
    for (int i = 0; i < aLength; i++) {
// If the sizes of the two dimensions are not equal, broadcasting is not possible
      if (a[i] != b[i] && a[i] != -1 && b[i] != -1) {
        return false;
      }
    }

// If all of the conditions are met, broadcasting is possible
    return true;
  }

  Tensor elementWiseMultiply(Tensor a, Tensor b) {
// Get the size of the smaller tensor
    int smallerSize = math.min(a.size, b.size);

// Create an empty tensor with the size of the smaller tensor
    Tensor result = Tensor.zeros([smallerSize]);

// Perform element-wise multiplication
    for (int i = 0; i < smallerSize; i++) {
      result._data[i] = a._data[i] * b._data[i];
    }

    return result;
  }

  List<int> broadcastWith(List<int> a, List<int> b) {
// Get the lengths of the two shapes
    int aLength = a.length;
    int bLength = b.length;

// Initialize the broadcasted shape
    List<int> broadcastedShape = [];

// Iterate over the dimensions of the two shapes
    for (int i = 0; i < aLength; i++) {
// If the dimensions are equal, add the dimension to the broadcasted shape
      if (a[i] == b[i]) {
        broadcastedShape.add(a[i]);
      }
// Otherwise, add the larger dimension to the broadcasted shape
      else {
        broadcastedShape.add(math.max(a[i], b[i]));
      }
    }

// Add any remaining dimensions from the larger shape to the broadcasted shape
    for (int i = aLength; i < bLength; i++) {
      broadcastedShape.add(b[i]);
    }

    return broadcastedShape;
  }

// Funções de ativação
  Tensor sigmoid() {
    List<double> result = List<double>.filled(size, 0);

    for (int i = 0; i < size; i++) {
      result[i] = 1 / (1 + math.exp(-_data[i]));
    }

    return createTensor(result, shape: _shape);
  }

  Tensor relu() {
    List<double> result = List<double>.filled(_data.length, 0);
// print('result ${_data.length}');
// print('result $_data');
    for (int i = 0; i < _data.length; i++) {
      result[i] = math.max(0, _data[i]);
    }
// print('result $result');

    Tensor newTensor = Tensor.createTensor(result, shape: _shape);
    return newTensor;
  }

  Tensor tanh() {
    List<double> result = List<double>.filled(size, 0);

    for (int i = 0; i < size; i++) {
      double expX = math.exp(_data[i]);
      double expMinusX = math.exp(-_data[i]);
      result[i] = (expX - expMinusX) / (expX + expMinusX);
    }

    Tensor newTensor = Tensor.createTensor(result, shape: _shape);
    return newTensor;
  }

// Métodos de redimensionamento

  Tensor reshape(List<int> newShape, {double fillValue = 0.0}) {
    List<double> data = _data;
    List<double> zeros = List.generate(_calculateSize(newShape), (_) => 0);

// Iterate over the new shape
    for (int i = 0;
        i < newShape.reduce((value, element) => value * element);
        i++) {
// If the index is within the range of the original data, add the value of the original data to the data list
      if (i < data.length) {
        data[i] = _data[i];
      } else {
// Otherwise, add a zero to the zeros list
        zeros[i] = 0;
      }
    }

// Combine the two lists
    List<double> mergedData = data + zeros;

// Create new tensor
    Tensor newTensor = Tensor.createTensor(mergedData, shape: newShape);

    return newTensor;
  }

  Tensor reshapeOutput(Tensor output, List<int> desiredOutputShape) {
    if (output.shape == desiredOutputShape) {
      return output;
    }

    List<double> newData =
        List.filled(desiredOutputShape.reduce((a, b) => a * b), 0);

    int originalHeight = output.shape[0];
    int originalWidth = output.shape[1];
    int originalChannels = output.shape[2];

    int desiredHeight = desiredOutputShape[0];
    int desiredWidth = desiredOutputShape[1];
    int desiredChannels = desiredOutputShape[2];

    double xRatio = originalWidth / desiredWidth;
    double yRatio = originalHeight / desiredHeight;

    for (int y = 0; y < desiredHeight; y++) {
      for (int x = 0; x < desiredWidth; x++) {
        double xFloat = x * xRatio;
        double yFloat = y * yRatio;
        int x1 = xFloat.toInt();
        int y1 = yFloat.toInt();
        int x2 = (xFloat + 1).toInt();
        int y2 = (yFloat + 1).toInt();

        x1 = x1.clamp(0, originalWidth - 1);
        x2 = x2.clamp(0, originalWidth - 1);
        y1 = y1.clamp(0, originalHeight - 1);
        y2 = y2.clamp(0, originalHeight - 1);

        double dx = xFloat - x1;
        double dy = yFloat - y1;

        for (int c = 0; c < desiredChannels; c++) {
          int index1 = (y1 * originalWidth + x1) * originalChannels + c;
          int index2 = (y1 * originalWidth + x2) * originalChannels + c;
          int index3 = (y2 * originalWidth + x1) * originalChannels + c;
          int index4 = (y2 * originalWidth + x2) * originalChannels + c;

          double w1 = (1 - dx) * (1 - dy);
          double w2 = dx * (1 - dy);
          double w3 = (1 - dx) * dy;
          double w4 = dx * dy;

          double interpolatedValue = output.getValue(index1) * w1 +
              output.getValue(index2) * w2 +
              output.getValue(index3) * w3 +
              output.getValue(index4) * w4;

          int newIndex = (y * desiredWidth + x) * desiredChannels + c;
          newData[newIndex] = interpolatedValue;
        }
      }
    }

    Tensor newOutput = Tensor.createTensor(newData, shape: desiredOutputShape);
    return newOutput;
  }

  // Tensor reshapeOutput(Tensor output, List<int> desiredOutputShape) {
  //   if (output.shape == desiredOutputShape) {
  //     return output;
  //   }

  //   List<double> newData =
  //       List.filled(desiredOutputShape.reduce((a, b) => a * b), 0);

  //   int originalHeight = output.shape[0];
  //   int originalWidth = output.shape[1];
  //   int desiredHeight = desiredOutputShape[0];
  //   int desiredWidth = desiredOutputShape[1];

  //   double xRatio = originalWidth / desiredWidth;
  //   double yRatio = originalHeight / desiredHeight;

  //   for (int y = 0; y < desiredHeight; y++) {
  //     for (int x = 0; x < desiredWidth; x++) {
  //       double xFloat = x * xRatio;
  //       double yFloat = y * yRatio;
  //       int x1 = xFloat.toInt();
  //       int y1 = yFloat.toInt();
  //       int x2 = (xFloat + 1).toInt();
  //       int y2 = (yFloat + 1).toInt();

  //       x1 = x1.clamp(0, originalWidth - 1);
  //       x2 = x2.clamp(0, originalWidth - 1);
  //       y1 = y1.clamp(0, originalHeight - 1);
  //       y2 = y2.clamp(0, originalHeight - 1);

  //       double dx = xFloat - x1;
  //       double dy = yFloat - y1;

  //       double w1 = (1 - dx) * (1 - dy);
  //       double w2 = dx * (1 - dy);
  //       double w3 = (1 - dx) * dy;
  //       double w4 = dx * dy;

  //       double interpolatedValue =
  //           output.getValue(y1 * originalWidth + x1) * w1 +
  //               output.getValue(y1 * originalWidth + x2) * w2 +
  //               output.getValue(y2 * originalWidth + x1) * w3 +
  //               output.getValue(y2 * originalWidth + x2) * w4;

  //       newData[y * desiredWidth + x] = interpolatedValue;
  //     }
  //   }
  //   Tensor newOutput = Tensor.createTensor(newData, shape: desiredOutputShape);
  //   return newOutput;
  // }

//   Tensor reshapeOutput(Tensor output, List<int> desiredOutputShape) {
// // Create a new tensor with the desired output shape
//     Tensor result = Tensor.zeros(desiredOutputShape);

// // Calculate the total number of elements in the output tensor
//     int totalOutputElements = result.size;

// // Iterate over the elements of the input tensor
//     int outputIndex = 0;
//     int outputShapeOrder = output.size ~/ totalOutputElements;
//     for (int i = 0; i < output.size; i++) {
// // print('i + outputShapeOrder * i ${i + outputShapeOrder * i}');
//       if (i >= totalOutputElements ||
//           i + outputShapeOrder * i > output.size - 1) {
//         print('break at $i ');
//         break; // Reached the limit of output elements
//       }

// // Assign the input element to the output tensor
//       result._data[outputIndex] = output._data[i + outputShapeOrder * i];

//       outputIndex++;
//     }

//     return result;
//   }

  List<int> _calculateStrides(List<int> shape) {
    List<int> strides = List<int>.filled(shape.length, 0);
    strides[strides.length - 1] = 1;
    for (int i = strides.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

// Serialização e desserialização
  Map<String, dynamic> toJSON() {
    return {};
  }

  void fromJSON(Map<String, dynamic> json) {
// Implementação padrão vazia
  }

  String serialize() {
    final jsonMap = toJSON();
    return jsonEncode(jsonMap);
  }

  static Tensor deserialize(String serializedData) {
    final jsonMap = jsonDecode(serializedData);
    final tensor = createTensor([], shape: []);
    tensor.fromJSON(jsonMap);
    return tensor;
  }

// Conversores
  static List<double> convertUint8ListToDoubleList(Uint8List uint8List) {
    return uint8List.map((value) => value.toDouble()).toList();
  }

  static Uint8List convertDoubleListToUint8List(List<double> doubleList) {
    return Uint8List.fromList(
        doubleList.map((value) => value.clamp(0.0, 255.0).toInt()).toList());
  }

  static List<double> convertUint8ListToDoubleListWithHeader(
      Uint8List uint8List) {
// Obtém o comprimento do cabeçalho
    int headerLength = uint8List[0];

// Obtém os dados da imagem
    Uint8List dataBuffer = uint8List.sublist(headerLength, uint8List.length);

// Converte os dados da imagem para um `List<double>`
    List<double> doubleList =
        dataBuffer.map((value) => value.toDouble()).toList();

    return doubleList;
  }

  static Uint8List? readPNGHeader(Uint8List imageData) {
// Leia os primeiros 8 bytes do Uint8List.
    List<int> magicBytes = imageData.sublist(0, 8);

// Se os bytes mágicos forem "137 80 78 71 13 10 26 10", então a imagem é um PNG.
    print("magicBytes $magicBytes");
    print("magicBytes ${magicBytes.length}");
    print("magicBytes ${magicBytes[0].runtimeType}");
    print("magicBytes ${magicBytes[1]}");
    print("magicBytes ${magicBytes[2]}");
    print("magicBytes ${magicBytes[3]}");
    print("magicBytes ${magicBytes[4]}");
    print("magicBytes ${magicBytes[5]}");
    print("magicBytes ${magicBytes[6]}");
    print("magicBytes ${magicBytes[7]}");
    if (magicBytes[0] != 137 &&
        magicBytes[1] != 80 &&
        magicBytes[2] != 78 &&
        magicBytes[3] != 71 &&
        magicBytes[4] != 13 &&
        magicBytes[5] != 10 &&
        magicBytes[6] != 26 &&
        magicBytes[7] != 10) {
      print("magicBytes null ${magicBytes[0]}");
      return null;
    }

// Leia os próximos 4 bytes.
    int chunkLength = imageData[16] | (imageData[17] << 8);
    print("int chunkLength $chunkLength");
// Calcule o tamanho do header.
    int headerLength = 8 + chunkLength;
    print("int headerLength $headerLength");

    return Uint8List.fromList(
        imageData.getRange(0, headerLength).toList()); // PNG
  }

  static SusiImageInfo createImageInfo(
      String type, int width, int height, int colorDepth) {
    return SusiImageInfo(type, width, height, colorDepth);
  }

  static Uint8List createUint8ListImage(
      List<double> doubleList, SusiImageInfo imageInfo) {
// Create header buffer with PNG signature
    Uint8List headerBuffer = Uint8List(17);
    headerBuffer[0] = 0x89;
    headerBuffer[1] = 0x50;
    headerBuffer[2] = 0x4E;
    headerBuffer[3] = 0x47;

// Populate IHDR chunk with image information
    switch (imageInfo.type) {
      case "PNG":
        headerBuffer[4] = imageInfo.width;
        headerBuffer[5] = (imageInfo.width >> 8) & 0xFF;
        headerBuffer[6] = (imageInfo.width >> 16) & 0xFF;

        headerBuffer[7] = (imageInfo.width >> 24) & 0xFF;

        headerBuffer[8] = imageInfo.height;
        headerBuffer[9] = (imageInfo.height >> 8) & 0xFF;
        headerBuffer[10] = (imageInfo.height >> 16) & 0xFF;
        headerBuffer[11] = (imageInfo.height >> 24) & 0xFF;

        headerBuffer[12] = imageInfo.colorDepth;
        headerBuffer[13] = 6; // Color type for PNG with RGB color space
        headerBuffer[14] = 0; // Compression method (0 for deflate)
        headerBuffer[15] = 0; // Filter method (0 for none)
        headerBuffer[16] = 0; // Interlace method (0 for no interlace)
        break;
      default:
        throw ArgumentError("Unsupported image type: ${imageInfo.type}");
    }

// Calculate checksum for IHDR chunk
    int checksum = 0;
    for (int i = 4; i < headerBuffer.length - 1; i++) {
      checksum ^= headerBuffer[i];
    }

    headerBuffer[headerBuffer.length - 1] = checksum;

// Create data buffer from double list
    Uint8List dataBuffer =
        Uint8List.fromList(doubleList.map((value) => value.toInt()).toList());

// Concatene header and data buffers
    Uint8List uint8List = concatenateUint8Lists(headerBuffer, dataBuffer);

    return uint8List;
  }

  static Uint8List concatenateUint8Lists(
      Uint8List headerBuffer, Uint8List dataBuffer) {
// Cria um novo `Uint8List` com o tamanho da soma dos dois `Uint8List`
    Uint8List uint8List = Uint8List(headerBuffer.length + dataBuffer.length);

// Copia os dados do `headerBuffer` para o novo `Uint8List`
    uint8List.setRange(0, headerBuffer.length, headerBuffer);

// Copia os dados do `dataBuffer` para o novo `Uint8List`
    uint8List.setRange(headerBuffer.length,
        headerBuffer.length + dataBuffer.length, dataBuffer);

    return uint8List;
  }
}

class PngHeaderAndData {
  final Uint8List data;
  final int width;
  final int height;
  final int colorMode;

  PngHeaderAndData({
    required this.data,
    required this.width,
    required this.height,
    required this.colorMode,
  });

  static PngHeaderAndData? getImageInfo(Uint8List data) {
    List<int> magicBytes = data.sublist(0, 7);

// Verifica se os magic bytes são válidos
    if (magicBytes[0] != 137 &&
        magicBytes[1] != 80 &&
        magicBytes[2] != 78 &&
        magicBytes[3] != 71 &&
        magicBytes[4] != 13 &&
        magicBytes[5] != 10 &&
        magicBytes[6] != 26 &&
        magicBytes[7] != 10) {
// Retorna null se os magic bytes forem inválidos
      print('magicBytes == null');
      return null;
    }

    int size = data.length;

// Obtém o corpo do IHDR, que tem 13 bytes de dados
// Uint8List body = data;
    Uint8List header = data.sublist(0, 55);
    Uint8List body = data.sublist(56, size - 12);
    Uint8List end = data.sublist(size - 11, size);

    int height = data[19];

    int width = data[23];

    int colorType = 0;

    if (data[25] == 0) {
      colorType = 1;
    } else if (data[25] == 2) {
      colorType = 3;
    } else if (data[25] == 3) {
      colorType = 1;
    } else if (data[25] == 4) {
      colorType = 2;
    } else if (data[25] == 6) {
      colorType = 4;
    }
    print('size body ${body.length}');

// Cria um objeto ImageInfo com as informações da imagem
    PngHeaderAndData imageInfo = PngHeaderAndData(
      height: height,
      width: width,
      colorMode: colorType,
      data: body,
    );
    print('imageInfo $imageInfo');

// Retorna o objeto ImageInfo
    return imageInfo;
  }

  static Uint8List newImage({
    required int height,
    required int width,
    required ByteBuffer bytes,
  }) {
    print('bytes ${bytes.lengthInBytes}');

// Cria uma imagem a partir do tensor
    img.Image image = img.Image.fromBytes(
      width: width,
      height: height,
      numChannels: 4,
      format: img.Format.uint8,
      bytes: bytes,
    );
    print('object');

// Cria um PngEncoder com as opções desejadas
    img.PngEncoder encoder = img.PngEncoder(
      level: 6, // Nível de compressão, de 0 a 9
      filter: img.PngFilter.paeth, // Tipo de filtragem, de 0 a 4
    );

// Cria um uint8list a partir da imagem
    Uint8List uint8list = encoder.encode(image);
    return uint8list;
  }
}
