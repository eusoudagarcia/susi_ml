import '../models/models.dart';
import '../models/tensors.dart';
import '../parallel/parallelism.dart';

void main() async {
  ModelBuilder modelBuilder = ModelBuilder();

// Enviesamentos
  Tensor biases = Tensor.zeros([32]);
// Pesos
  Tensor weights = biases.random([3, 3, 1, 32]);

// Camada Conv2D
  modelBuilder.addLayer('Conv2D', {
    'filters': 32,
    'kernelSize': [3, 3],
    'activation': ActivationFunction.relu,
    'weights': weights,
    'biases': biases,
    'desiredOutputShape': [28, 28, 32],
  });

// Camada MaxPooling2D
  modelBuilder.addLayer('MaxPooling2D', {
    'poolSize': [2, 2],
  });

// Camada Dense
  modelBuilder.addLayer('Dense', {
    'neurons': 1024,
    'activation': ActivationFunction.relu,
  });

// Camada Dropout
  modelBuilder.addLayer('Dropout', {
    'rate': 0.2,
  });

// Camada Conv2DTranspose
  modelBuilder.addLayer('Conv2DTranspose', {
    'filters': 1,
    'kernelSize': [3, 3],
    'activation': ActivationFunction.sigmoid,
  });

// Construir o modelo
  Model model = modelBuilder.buildModel();

  ParallelismImpl parallelism = ParallelismImpl();

  int numeric = 5;
  int numeric2 = 20;

  Future<Model> modelBuilded = parallelism
      .execute<Model>(() => modelBuilder.buildModel(), numberOfCores: 2);

  print(await modelBuilded);

  Model model2 = await modelBuilded;
  model2.summary();

  // Exemplo 1: Modelo com 3 camadas (Conv2D, MaxPooling2D e Dense)
  // ModelBuilder modelBuilder1 = ModelBuilder();

  // modelBuilder1.addLayer('Conv2D', {
  //   'filters': 32,
  //   'kernelSize': [3, 3],
  //   'activation': ActivationFunction.relu,
  // });
  // modelBuilder1
  //     .addLayer('Dense', {'units': 32, 'activation': ActivationFunction.tanh});
  // modelBuilder1.addLayer('MaxPooling2D', {
  //   'poolSize': [2, 2]
  // });
  // modelBuilder1
  //     .addLayer('Dense', {'units': 64, 'activation': ActivationFunction.relu});
  // Model model1 = modelBuilder1.buildModel();

  // // Exemplo 2: Modelo com 4 camadas (Conv2D, Conv2D, Flatten e Dense)
  // ModelBuilder modelBuilder2 = ModelBuilder();
  // modelBuilder2.addLayer('Conv2D', {
  //   'filters': 64,
  //   'kernelSize': [3, 3],
  //   'activation': ActivationFunction.relu
  // });
  // modelBuilder2.addLayer('Conv2D', {
  //   'filters': 128,
  //   'kernelSize': [3, 3],
  //   'activation': ActivationFunction.relu
  // });
  // modelBuilder2.addLayer('Flatten', {});
  // modelBuilder2
  //     .addLayer('Dense', {'units': 10, 'activation': ActivationFunction.relu});
  // Model model2 = modelBuilder2.buildModel();

  // // Exemplo 3: Modelo com 2 camadas (Conv2D e Dense)
  // ModelBuilder modelBuilder3 = ModelBuilder();
  // modelBuilder3.addLayer('Conv2D', {
  //   'filters': 16,
  //   'kernelSize': [5, 5],
  //   'activation': ActivationFunction.relu
  // });
  // modelBuilder3
  //     .addLayer('Dense', {'units': 32, 'activation': ActivationFunction.relu});
  // Model model3 = modelBuilder3.buildModel();
  // print('Model 1 ---------');
  // model1.summary();
  // print('Model 1 ---------');
  // print('Model 2 ---------');
  // model2.summary();
  // print('Model 2 ---------');
  // print('Model 3 ---------');
  // model3.summary();
  // print('Model 3 ---------');

  // Utilize os modelos construídos para treinar e avaliar sua performance
}

// import 'package:susi_ml/models/tensors.dart';
// import 'package:susi_ml/network/cnn_trainer.dart';
// import 'package:susi_ml/parallel/parallelism.dart';

// void main() async {
//   // Criar o tensor de exemplo
//   Tensor tensor = Tensor.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

//   // Sem paralelismo
//   Stopwatch stopwatch = Stopwatch()..start();
//   Tensor resultWithoutParallelism = CNNTrainer.preprocessData(
//     tensor: tensor,
//     newSize: 8,
//     normalize: true,
//   );
//   print('Result without parallelism: ${resultWithoutParallelism.data}');
//   print('Elapsed time without parallelism: ${stopwatch.elapsed}');

//   // Com paralelismo
//   stopwatch.reset();
//   Parallelism parallelism = ParallelismImpl();
//   Future<Tensor> futureResultWithParallelism = parallelism.execute<Tensor>(() {
//     return CNNTrainer.preprocessData(
//       tensor: tensor,
//       newSize: 8,
//       normalize: true,
//     );
//   });
//   Tensor resultWithParallelism = await futureResultWithParallelism;
//   print('Result with parallelism: ${resultWithParallelism.data}');
//   print('Elapsed time with parallelism: ${stopwatch.elapsed}');
// }

// import 'package:susi_ml/models/tensors.dart';

// void main() {
//   Stopwatch stopwatch = Stopwatch()..start();

//   // Create example tensors
//   Tensor tensorA = Tensor.fromList([1, 2, 3]);
//   Tensor tensorB = Tensor.fromList([4, 5, 6]);

//   // Perform operations without parallelism
//   Tensor resultA = tensorA + tensorB;
//   Tensor resultB = tensorA * tensorB;
//   Tensor resultC = tensorA.relu();
//   Tensor resultD = tensorB.tanh();

//   // Print the results
//   print('Result A (Sum): ${resultA.data}');
//   print('Result B (Multiplication): ${resultB.data}');
//   print('Result C (ReLU): ${resultC.data}');
//   print('Result D (Tanh): ${resultD.data}');

//   stopwatch.stop(); // Stop the stopwatch
//   print('Algorithm executed in ${stopwatch.elapsed}');
// }

// import 'package:susi_ml/parallel/parallelism.dart';

// import '../models/tensors.dart';

// void main() async {
//   Stopwatch stopwatch = Stopwatch()..start();
//   // Criar os tensores de exemplo
//   Tensor tensorA = Tensor.fromList([1, 2, 3]);
//   Tensor tensorB = Tensor.fromList([4, 5, 6]);

//   // Instanciar a classe Paralelism
//   Parallelism paralelism = ParallelismImpl();

//   // Executar as operações em paralelo usando o Paralelism
//   Future<Tensor> futureResultA = paralelism
//       .execute<Tensor>(() => tensorA + tensorB + tensorB + tensorB + tensorB);
//   Future<Tensor> futureResultB =
//       paralelism.execute<Tensor>(() => tensorA * tensorB);
//   Future<Tensor> futureResultC =
//       paralelism.execute<Tensor>(() => tensorA.relu());
//   Future<Tensor> futureResultD =
//       paralelism.execute<Tensor>(() => tensorB.tanh());

//   // Aguardar os resultados
//   Tensor resultA = await futureResultA;
//   Tensor resultB = await futureResultB;
//   Tensor resultC = await futureResultC;
//   Tensor resultD = await futureResultD;

//   // Imprimir os resultados
//   print('Result A (Soma): ${resultA.data}');
//   print('Result B (Multiplicação): ${resultB.data}');
//   print('Result C (ReLU): ${resultC.data}');
//   print('Result D (Tanh): ${resultD.data}');
//   print('algoritmo executado em ${stopwatch.elapsed}');
// }

// import 'dart:io';
// import 'dart:isolate';

// void main() async {
//   Stopwatch stopwatch = Stopwatch()..start(); // inicia o cronômetro
//   print(Platform.numberOfProcessors);
//   List<int> numbersToCalculate = [24, 29, 34, 35, 39, 49, 14500];
//   ReceivePort receivePort = ReceivePort();

//   for (int number in numbersToCalculate) {
//     // cria um novo isolado para cada número da sequência que você quer calcular
//     await Isolate.spawn(fibonacci, [number, receivePort.sendPort]);
//   }

//   // fecha o receive port
//   receivePort.close();

//   stopwatch.stop(); // para o cronômetro
//   print(
//       'fibonacci executado em ${stopwatch.elapsed}'); // imprime a duração total da operação
// }

// // a função que será executada pelo isolado
// void fibonacci(List args) {
//   // extrai o argumento n e o send port da lista de argumentos
//   int n = args[0];
//   SendPort sendPort = args[1];
//   // calcula o n-ésimo termo da sequência de fibonacci
//   int result = 1;
//   int prev = 0;
//   for (int i = 0; i < n; i++) {
//     int temp = result;
//     result = result + prev;
//     prev = temp;
//   }
//   // envia o resultado para o send port do main
//   print("$n $result");
//   sendPort.send(result);
// }

// import 'dart:isolate';

// void main() async {
//   Stopwatch stopwatch = Stopwatch()..start(); // inicia o cronômetro
//   // cria um receive port para receber mensagens dos isolados
//   ReceivePort receivePort = ReceivePort();
//   // cria uma lista de futures para armazenar os resultados dos isolados
//   List<Future> futures = [];
//   // cria um novo isolado para cada sequência que você quer calcular
//   futures.add(Isolate.spawn(fibonacci, [24, receivePort.sendPort]));
//   futures.add(Isolate.spawn(fibonacci, [29, receivePort.sendPort]));
//   futures.add(Isolate.spawn(fibonacci, [34, receivePort.sendPort]));
//   futures.add(Isolate.spawn(fibonacci, [35, receivePort.sendPort]));
//   futures.add(Isolate.spawn(fibonacci, [39, receivePort.sendPort]));
//   futures.add(Isolate.spawn(fibonacci, [40, receivePort.sendPort]));
//   futures.add(Isolate.spawn(fibonacci, [41, receivePort.sendPort]));
//   futures.add(Isolate.spawn(fibonacci, [42, receivePort.sendPort]));
//   futures.add(Isolate.spawn(fibonacci, [49, receivePort.sendPort]));
//   // futures.add(Isolate.spawn(fibonacci, [200000, receivePort.sendPort]));
//   // espera por todos os isolados terminarem

//   // fecha o receive port
//   receivePort.close();
//   stopwatch.stop(); // para o cronômetro
//   print(
//       'fibonacci executado em ${stopwatch.elapsed}'); // imprime a duração total da operação
// }

// // a função que será executada pelo isolado
// void fibonacci(List args) {
//   // extrai o argumento n e o send port da lista de argumentos
//   int n = args[0];
//   SendPort sendPort = args[1];
//   // calcula o n-ésimo termo da sequência de fibonacci
//   int result = 1;
//   int prev = 0;
//   for (int i = 0; i < n; i++) {
//     int temp = result;
//     result = result + prev;
//     prev = temp;
//   }
//   // envia o resultado para o send port do main
//   print("$n $result");
//   sendPort.send(result);
// }

// import 'dart:io';

// void main() {
//   Stopwatch stopwatch = Stopwatch()..start();
//   print(Platform.numberOfProcessors);
//   print(fibonacci(25)); // 75025
//   print(fibonacci(30)); // 832040
//   print(fibonacci(35)); // 9227465
//   print(fibonacci(36)); // 14930352
//   print(fibonacci(40)); // 102334155
//   // print(fibonacci(41));
//   // print(fibonacci(42));
//   // print(fibonacci(43));
//   // print(fibonacci(50));
//   stopwatch.stop();
//   print('fibonacci(40) executado em ${stopwatch.elapsed}');
// }

// int fibonacci(int n) {
//   if (n <= 1) {
//     return n;
//   } else {
//     return fibonacci(n - 1) + fibonacci(n - 2);
//   }
// }
