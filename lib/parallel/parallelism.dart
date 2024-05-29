// import 'dart:async';
// import 'dart:isolate';

// abstract class Parallelism {
//   Future<T> execute<T>(Function function);
// }

// class ParallelismImpl implements Parallelism {
//   @override
//   Future<T> execute<T>(Function function) {
//     Completer<T> completer = Completer<T>();
//     ReceivePort receivePort = ReceivePort();

//     Isolate.spawn(_executeIsolate, [function, receivePort.sendPort])
//         .then((isolate) {
//       receivePort.listen((message) {
//         if (message is T) {
//           completer.complete(message);
//           receivePort.close();
//           isolate.kill(priority: Isolate.immediate);
//         }
//       });
//     });

//     return completer.future;
//   }

//   static void _executeIsolate(List<Object> args) {
//     Function? function;
//     SendPort? sendPort;

//     if (args.length >= 2) {
//       function = args[0] as Function;
//       sendPort = args[1] as SendPort;
//     }

//     if (function != null && sendPort != null) {
//       dynamic result = function();
//       sendPort.send(result);
//     }
//   }
// }

import 'dart:async';
import 'dart:isolate';

abstract class Parallelism {
  Future<T> execute<T>(Function function);
}

class ParallelismImpl implements Parallelism {
  @override
  Future<T> execute<T>(Function function, {int numberOfCores = 1}) async {
    if (numberOfCores < 1) {
      throw ArgumentError(
          "The number of cores must be greater than or equal to 1.");
    }

    Completer<T> completer = Completer<T>();
    ReceivePort receivePort = ReceivePort();
    Isolate? isolate;

    for (int i = 0; i < numberOfCores; i++) {
      isolate = await Isolate.spawn(
          _executeIsolate, [function, receivePort.sendPort]);
    }

    receivePort.listen((message) async {
      if (message is T) {
        completer.complete(message);
        receivePort.close();
        isolate!.kill(priority: Isolate.immediate);
      } else {
        // Handle other types of messages here.
      }
    });

    return completer.future;
  }

  static void _executeIsolate(List<Object> args) {
    Function? function;
    SendPort? sendPort;

    if (args.length >= 2) {
      function = args[0] as Function;
      sendPort = args[1] as SendPort;
    }

    if (function != null && sendPort != null) {
      dynamic result = function();
      sendPort.send(result);
    }
  }
}
