import 'package:susi_ml/models/tensors.dart';

enum ActivationFunction {
  relu,
  tanh,
  sigmoid,
}

class ModelBuilder {
  late List<Map<String, dynamic>> _layers;
  late String _optimizer;
  late String _loss;
  late List<String> _metrics;

  ModelBuilder() {
    _layers = [];
    _optimizer = 'adam';
    _loss = 'categorical_crossentropy';
    _metrics = ['accuracy'];
  }

  void addLayer(String layerType, Map<String, dynamic> parameters) {
    _layers.add({'layerType': layerType, 'parameters': parameters});
  }

  void setOptimizer(String optimizer) {
    _optimizer = optimizer;
  }

  void setLoss(String loss) {
    _loss = loss;
  }

  void setMetrics(List<String> metrics) {
    _metrics = metrics;
  }

  Model buildModel() {
    Model model = Model();
    for (var layerData in _layers) {
      String layerType = layerData['layerType'];
      Map<String, dynamic> parameters = layerData['parameters'];

      if (layerType == 'Conv2D') {
        int filters = parameters['filters'];
        List<int> kernelSize = parameters['kernelSize'];
        ActivationFunction activation = parameters['activation'];
        Tensor? weights = parameters['weights'];
        Tensor? biases = parameters['biases'];
        List<int> desiredOutputShape = parameters['desiredOutputShape'];

// Adicionar camada Conv2D ao modelo com os parâmetros fornecidos
        model.add(Conv2D(
          filters: filters,
          kernelSize: kernelSize,
          activation: activation,
          weights: weights,
          biases: biases,
          desiredOutputShape: desiredOutputShape,
        ));
      } else if (layerType == 'MaxPooling2D') {
        List<int> poolSize = parameters['poolSize'];
// Adicionar camada MaxPooling2D ao modelo com os parâmetros fornecidos
        model.add(MaxPooling2D(poolSize: poolSize));
      } else if (layerType == 'Flatten') {
// Adicionar camada Flatten ao modelo
        model.add(Flatten());
      } else if (layerType == 'Dense') {
        int units = parameters['units'];
        ActivationFunction activation = parameters['activation'];
// Adicionar camada Dense ao modelo com os parâmetros fornecidos
        model.add(Dense(units: units, activation: activation));
      }
    }

// Configurar otimizador, função de perda e métricas do modelo
    model.compile(optimizer: _optimizer, loss: _loss, metrics: _metrics);

    return model;
  }
}

class Model {
  List<Layer>? layers;
  String? optimizer;
  String? loss;
  List<String>? metrics;

  Model() {
    layers = [];
    optimizer = 'adam';
    loss = 'categorical_crossentropy';
    metrics = ['accuracy'];
  }

  void add(Layer layer) {
    layers!.add(layer);
  }

  void compile({String? optimizer, String? loss, List<String>? metrics}) {
    if (optimizer != null) {
      this.optimizer = optimizer;
    }
    if (loss != null) {
      this.loss = loss;
    }
    if (metrics != null) {
      this.metrics = metrics;
    }
  }

  void summary() {
    print('Model Summary');
    print('Optimizer: $optimizer');
    print('Loss: $loss');
    print('Metrics: $metrics');
    print('Layers:');
    for (var i = 0; i < layers!.length; i++) {
      print('Layer ${i + 1}: ${layers![i].toString()}');
    }
  }

  void trainOnBatch(Tensor x, Tensor y) async {
// Forward pass
    Tensor output = x;

    for (var layer in layers!) {
      output = await layer.forward(output);
    }

// Backward pass
    Tensor lossGradient = output - y;

    for (var i = layers!.length - 1; i >= 0; i--) {
      lossGradient = layers![i].backward(lossGradient, output, x);
    }
  }
}

abstract class Layer {
  String layerType;
  Map<String, dynamic> parameters;

  Layer({required this.layerType, this.parameters = const {}});

  Future<Tensor> forward(Tensor input);

  Tensor backward(Tensor inputGradient, Tensor output, Tensor input);

  @override
  String toString() {
    return '$layerType\nParameters: $parameters';
  }
}

class Conv2D extends Layer {
  Conv2D({
    required int filters,
    required List<int> kernelSize,
    ActivationFunction activation = ActivationFunction.relu,
    Tensor? weights,
    Tensor? biases,
    List<int>? desiredOutputShape,
  }) : super(layerType: "Conv2D", parameters: {
          "filters": filters,
          "kernelSize": kernelSize,
          "activation": activation,
          "weights": weights,
          "biases": biases,
          'desiredOutputShape': desiredOutputShape,
        });

  @override
  Future<Tensor> forward(Tensor input) async {
    int filters = parameters['filters'];
    List<int> kernelSize = parameters['kernelSize'];
    ActivationFunction activation = parameters['activation'];
    Tensor? weights = parameters['weights'];
    Tensor? biases = parameters['biases'];
    List<int>? desiredOutputShape = parameters['desiredOutputShape'];

// Implementação da camada Conv2D
// Processar a entrada e retornar a saída
// Calcule a saída da convolução

// print('input data ${input.data} shape ${input.shape}, size ${input.size}');
    Tensor output = await input.convolve2d(
      kernelSize,
      filters,
      activation,
      weights,
      biases,
    );
// print("output shape ${output.data}");
    print("output shape ${output.shape}");
// print("output size ${output.size}");
// print("desiredOutputShape $desiredOutputShape");

// Remodele a saída
    if (desiredOutputShape != null) {
      if (output.shape != desiredOutputShape) {
        output = output.reshapeOutput(output, desiredOutputShape);
      }
    }
// print("output shape ${output.shape}");
// print("output size ${output.size}");

    return output;
// Retornar a saída da camada
  }

  @override
  Tensor backward(Tensor inputGradient, Tensor output, Tensor input) {
// Implementação inicial da retropropagação da camada Conv2D
// Processar o gradiente de entrada e retornar o gradiente de saída
// ...

    return Tensor.createTensor([], shape: []); // Retornar o gradiente de saída
  }
}

class MaxPooling2D extends Layer {
  MaxPooling2D({required List<int> poolSize})
      : super(layerType: "MaxPooling2D", parameters: {
          "poolSize": poolSize,
        });

  @override
  Future<Tensor> forward(Tensor input) async {
    List<int> poolSize = parameters['poolSize'];

// Aplicar o max pooling
    Tensor output = input.maxPooling2d(input, poolSize);
// print("MaxPooling2D output ${output.data} shape ${output.shape}");

    return output; // Retornar a saída da camada
  }

  @override
  Tensor backward(Tensor inputGradient, Tensor output, Tensor input) {
// Implementação inicial da retropropagação da camada MaxPooling2D
// Processar o gradiente de entrada e retornar o gradiente de saída
// ...

    return Tensor.createTensor([], shape: []); // Retornar o gradiente de saída
  }
}

class Flatten extends Layer {
  Flatten() : super(layerType: "Flatten");

  @override
  Future<Tensor> forward(Tensor input) async {
// Implementação da camada Flatten
// Processar a entrada e retornar a saída
// ...

    return Tensor.createTensor([], shape: []); // Retornar a saída da camada
  }

  @override
  Tensor backward(Tensor inputGradient, Tensor output, Tensor input) {
// Implementação inicial da retropropagação da camada Flatten
// Processar o gradiente de entrada e retornar o gradiente de saída
// ...

    return Tensor.createTensor([], shape: []); // Retornar o gradiente de saída
  }
}

class Dense extends Layer {
  Dense({
    required int units,
    ActivationFunction activation = ActivationFunction.relu,
  }) : super(layerType: "Dense", parameters: {
          "units": units,
          "activation": activation,
        });

  @override
  Future<Tensor> forward(Tensor input) async {
    int units = parameters['units'];
    ActivationFunction activation = parameters['activation'];
    Tensor weights = parameters['weights'];
    Tensor biases = parameters['biases'];

// Implementação da camada Dense
// Processar a entrada e retornar a saída
// ...

    switch (activation) {
      case ActivationFunction.relu:
        input = input.relu();
        break;
      case ActivationFunction.tanh:
        input = input.tanh();
        break;
      case ActivationFunction.sigmoid:
        input = input.sigmoid();
// * Parallismo caso precise de exclusividade e velocidade no processamento
// ParallelismImpl parallelism = ParallelismImpl();
// Future<Tensor> inpuntParallelism =

//     parallelism.execute<Tensor>(() =>

//     input.sigmoid()

//     );

// inpuntParallelism;
        break;
      default:
        break;
    }

    return Tensor.createTensor([], shape: []); // Retornar a saída da camada
  }

  @override
  Tensor backward(Tensor inputGradient, Tensor output, Tensor input) {
// Implementação inicial da retropropagação da camada Dense
// Processar o gradiente de entrada e retornar o gradiente de saída
// ...

    return Tensor.createTensor([], shape: []); // Retornar o gradiente de saída
  }
}

class LSTM extends Layer {
  LSTM({
    required int units,
    required List<int> kernelSize,
    ActivationFunction activation = ActivationFunction.relu,
    int returnSequences = 0,
  }) : super(layerType: "LSTM", parameters: {
          "units": units,
          "kernelSize": kernelSize,
          "activation": activation,
          "returnSequences": returnSequences,
        });

// Implement the forward propagation method
  @override
  Future<Tensor> forward(Tensor input) async {
    int units = parameters['units'];
    List<int> kernelSize = parameters['kernelSize'];
    ActivationFunction activation = parameters['activation'];
    int returnSequences = parameters['returnSequences'];

// Process the input and return the output
// ...

    return Tensor.createTensor([], shape: []); // Return the output of the layer
  }

// Implement the backward propagation method
  @override
  Tensor backward(Tensor inputGradient, Tensor output, Tensor input) {
    int units = parameters['units'];
    List<int> kernelSize = parameters['kernelSize'];
    ActivationFunction activation = parameters['activation'];
    int returnSequences = parameters['returnSequences'];

// Process the input gradient and return the output gradient
// ...

    return Tensor.createTensor([], shape: []); // Return the output gradient
  }
}

class Conv2DTranspose extends Layer {
  Conv2DTranspose({
    required int filters,
    required List<int> kernelSize,
    required int strides,
    ActivationFunction activation = ActivationFunction.relu,
  }) : super(layerType: "Conv2DTranspose", parameters: {
          "filters": filters,
          "kernelSize": kernelSize,
          "strides": strides,
          "activation": activation,
        });

// Implement the forward propagation method
  @override
  Future<Tensor> forward(Tensor input) async {
    int filters = parameters['filters'];
    List<int> kernelSize = parameters['kernelSize'];
    int strides = parameters['strides'];
    ActivationFunction activation = parameters['activation'];

// Process the input and return the output
// ...

    return Tensor.createTensor([], shape: []); // Return the output of the layer
  }

// Implement the backward propagation method
  @override
  Tensor backward(Tensor inputGradient, Tensor output, Tensor input) {
    int filters = parameters['filters'];
    List<int> kernelSize = parameters['kernelSize'];
    int strides = parameters['strides'];
    ActivationFunction activation = parameters['activation'];

// Process the input gradient and return the output gradient
// ...

    return Tensor.createTensor([], shape: []); // Return the output gradient
  }
}

class TimeDistributed extends Layer {
  TimeDistributed({required Layer layer})
      : super(layerType: "TimeDistributed", parameters: {"layer": layer});

// Implement the forward propagation method
  @override
  Future<Tensor> forward(Tensor input) async {
    Layer layer = parameters['layer'];

// Process the input and return the output
// ...

    return Tensor.createTensor([], shape: []); // Return the output of the layer
  }

// Implement the backward propagation method
  @override
  Tensor backward(Tensor inputGradient, Tensor output, Tensor input) {
    Layer layer = parameters['layer'];

// Process the input gradient and return the output gradient
// ...

    return Tensor.createTensor([], shape: []); // Return the output gradient
  }
}
