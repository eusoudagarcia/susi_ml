import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../models/models.dart';
import '../models/tensors.dart';

void main() async {
  Directory currentDirectory = Directory.current;

  // String image1 = '1';
  // String image2 = '2';
  // String image3 = '3';
  // String image4 = '4';

  String image1 = '14471433500_cdaa22e3ea_m';

  print(
      "currentDirectory + image  $currentDirectory/lib/tests/flower_photos/daisy/$image1.jpg");
  File image = File(
      "${currentDirectory.path}/lib/tests/flower_photos/daisy/$image1.jpg");

  Uint8List? data = await image.readAsBytes();

  img.Image? newImage = img.decodeJpg(data);

  print("newImage format ${newImage!.format}");
  print("newImage channels ${newImage.extraChannels}");
  print("newImage bitsChannel ${newImage.bitsPerChannel}");
  print("newImage frameIndex ${newImage.frameIndex}");
  print("newImage frames ${newImage.frames}");
  print("newImage iccProfile ${newImage.iccProfile}");
  print("newImage numChannels ${newImage.numChannels}");
  print("newImage rowStride ${newImage.rowStride}");
  print("newImage w ${newImage.width}");
  print("newImage h ${newImage.height}");
  print("newImage pallet ${newImage.palette}");

  // Inicializa os pesos e enviesamentos
  Tensor biases =
      Tensor.zeros([newImage.width, newImage.height, newImage.numChannels]);
  // print('biases data ${biases.data} shape ${biases.shape}');
  Tensor weights = biases.random(
    [newImage.width, newImage.height, newImage.numChannels],
  );
  // print('weights data ${weights.data} shape ${weights.shape}');

  Conv2D conv2d = Conv2D(
    filters: 32,
    kernelSize: [10, 10],
    activation: ActivationFunction.relu,
    desiredOutputShape: [100, 100, 3],
    weights: weights,
    biases: biases,
  );

  // // Cria uma matriz de entrada
  // Tensor input = Tensor.zeros([100, 100, 3]);

  List<double> list =
      Tensor.convertUint8ListToDoubleList(newImage.buffer.asUint8List());

  Tensor input = Tensor.fromList(list,
      shape: [newImage.width, newImage.height, newImage.numChannels]);

  // Aplica a camada Conv2D
  Tensor output = await conv2d.forward(input);

  // print(output.data);

  Uint8List imageDataOutput = Tensor.convertDoubleListToUint8List(output.data);

  img.Image? otherImage = img.Image.fromBytes(
    width: 100,
    height: 100,
    bytes: imageDataOutput.buffer,
    numChannels: 3,
  );

  final png = img.encodePng(otherImage);
  await File('${currentDirectory.path}/lib/tests/image.png').writeAsBytes(png);
  // // Imprime a saída da camada
  // print("shape ${output.shape}");

  // print('output conv2d ${output.data}');

  // MaxPooling2D maxPooling2D = MaxPooling2D(poolSize: [2, 2]);

  // Tensor maxPooling2DOutput = maxPooling2D.forward(output);

  // print(
  //     "maxPooling2DOutput.data ${maxPooling2DOutput.data} shape ${maxPooling2DOutput.shape}");

  // Dense dense = Dense(units: 1024, activation: ActivationFunction.relu);

  // Tensor denseOutput = dense.forward(maxPooling2DOutput);
}
