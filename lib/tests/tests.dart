// Importar o pacote dart:test
// Importar o seu código
import 'package:susi_ml/models/tensors.dart';
import 'package:susi_ml/network/cnn_trainer.dart';
import 'package:test/test.dart';

// Definir uma função auxiliar para comparar dois Tensors
bool compareTensors(Tensor a, Tensor b) {
  if (a.size != b.size) {
    return false;
  }
  for (int i = 0; i < a.size; i++) {
    if (a.getValue(i) != b.getValue(i)) {
      return false;
    }
  }
  return true;
}

// Definir os testes
void main() {
  // Testar a função preprocessData com diferentes parâmetros
  test('preprocessData with resize', () {
    // Criar um Tensor de exemplo
    Tensor tensor = Tensor.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Chamar a função preprocessData com o método de redimensionamento
    Tensor result = CNNTrainer.preprocessData(
      tensor: tensor,
      newSize: 4,
      removeLasts: false,
    );

    // Verificar se o resultado é igual ao esperado
    expect(compareTensors(result, Tensor.fromList([3.0, 4.0, 5.0, 6.0])), true);
  });

  test('preprocessData with resize', () {
    // Criar um Tensor de exemplo
    Tensor tensor = Tensor.fromList([1, 2, 3, 4, 5, 6]);

    // Chamar a função preprocessData com o método de corte
    Tensor result = CNNTrainer.preprocessData(
      tensor: tensor,
      newSize: 8,
      addLasts: true,
    );

    // Verificar se o resultado é igual ao esperado
    expect(compareTensors(result, Tensor.fromList([1, 2, 3, 4, 5, 6, 0, 0])),
        true);
  });

  test('preprocessData with resize', () {
    // Criar um Tensor de exemplo
    Tensor tensor = Tensor.fromList([1, 2, 3, 4, 5, 6]);

    // Chamar a função preprocessData com o método de preenchimento
    Tensor result =
        CNNTrainer.preprocessData(tensor: tensor, removeLasts: true);

    // Verificar se o resultado é igual ao esperado
    expect(compareTensors(result, Tensor.fromList([1, 2, 3, 4, 5, 6])), true);
  });

// Testar outros parâmetros da função preprocessData
}
