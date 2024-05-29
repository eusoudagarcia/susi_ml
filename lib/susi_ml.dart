import 'package:susi_ml/models/tensors.dart';
import 'package:susi_ml/network/cnn_trainer.dart';

int calculate() {
  return 6 * 7;
}

void main() {
  print(calculate());
  print('hello world! susi_ml');

  // Criação de tensores de exemplo
  Tensor tensorA = Tensor.constant([1, 2, 2], 1.2);
  Tensor tensorB =
      tensorA.random([1, 2, 2], distribution: RandomDistribution.normal);

  Tensor preprocessed = CNNTrainer.preprocessData(
    tensor: tensorB,
    normalize: true,
  );
  print(preprocessed.data);
  // Exibição dos resultados
  print('Tensor A: ${tensorA.data}');
  print('Tensor B: ${tensorB.data}');

  // Operações matemáticas
  Tensor tensorSum = tensorA + tensorB;
  Tensor tensorProduct = tensorA * tensorB;
  print('Tensor B: ${tensorB.data}');
  tensorB.applyScalarOperation(scalar: 30);
  print('Tensor B: ${tensorB.data}');
  Tensor tanh = tensorB.transpose();
  Tensor relu = tensorB.resize(2);

  print('Tensor B: ${tanh.data}');
  print('Tensor B: ${relu.data}');
  print('Tensor B: ${relu.getValue(2)}');

  print('Tensor reluB: ${relu.data}');
  Tensor reluSigmoidal = relu.sigmoid();
  print('Tensor reluSigmoidal: ${reluSigmoidal.data}');
  print('Tensor reluSigmoidal: ${reluSigmoidal.normL2()}');
  print('Tensor normB: ${relu.normL2()}');

  print('Tensor normIB: ${relu.normInfinite()}');

  Tensor tensorNormalized = Tensor.constant([1, 2, 2], relu.normL2());
  print('Tensor norm: ${tensorNormalized.data}');
}
