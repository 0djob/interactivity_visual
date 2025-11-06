import 'package:flutter/material.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'dart:convert';
import 'dart:async';
import 'dart:math' as math;
import 'package:http/http.dart' as http;

void main() => runApp(const BehaviorSpaceApp());


class BehaviorSpaceApp extends StatelessWidget {
  const BehaviorSpaceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Behavior Space Visualization',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        useMaterial3: true,
      ),
      home: const BehaviorSpaceView(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class BehaviorSpaceView extends StatefulWidget {
  const BehaviorSpaceView({super.key});

  @override
  State<BehaviorSpaceView> createState() => _BehaviorSpaceViewState();
}

class _BehaviorSpaceViewState extends State<BehaviorSpaceView> {
  WebSocketChannel? channel;
  StreamSubscription? channelSubscription;
  bool isConnected = false;
  bool isPaused = false;
  bool _isConnecting = false;
  bool _shouldReconnect = true;
  Timer? _reconnectTimer;
  int _reconnectAttempts = 0;
  static const int _maxReconnectDelay = 30; // Max 30 seconds between attempts

  double iScore = 0.0;
  double conditionalComplexity = 0.0;
  double semiconditionalComplexity = 0.0;
  List<double> iScoreHistory = [];
  
  List<double> behaviorVector = [];
  List<double> oldPrediction = [];
  List<double> currentPrediction = [];
  
  double distanceOldToActual = 0.0;
  double distanceCurrentToActual = 0.0;
  double distanceDifference = 0.0;
  
  List<List<double>> trajectory2D = [];
  List<double>? oldPrediction2D;
  List<double>? currentPrediction2D;
  List<double>? actualPosition2D;
  bool pcaReady = false;
  
  int step = 0;

  @override
  void initState() {
    super.initState();
    _shouldReconnect = true;
    connectWebSocket();
  }

  /// Calculate exponential backoff delay for reconnection attempts
  int _getReconnectDelay() {
    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (max)
    final delay = math.min(
      math.pow(2, _reconnectAttempts).toInt(),
      _maxReconnectDelay,
    );
    return delay;
  }

  void connectWebSocket() {
    // Prevent multiple simultaneous connection attempts
    if (_isConnecting || !_shouldReconnect) {
      print('Connection attempt skipped (already connecting or should not reconnect)');
      return;
    }
    
    _isConnecting = true;
    _reconnectAttempts++;
    
    print('🔄 Attempting WebSocket connection (attempt #$_reconnectAttempts)...');

    try {
      // Close any existing connection first
      _cleanupConnection();

      channel = WebSocketChannel.connect(
        Uri.parse('wss://interactivity-agent.onrender.com/ws/agent'),
      );

      // Handle connection ready
      channel!.ready.then((_) {
        if (mounted && _shouldReconnect) {
          setState(() {
            isConnected = true;
            _isConnecting = false;
            _reconnectAttempts = 0; // Reset on successful connection
          });
          print('✅ Connected to behavior space agent');
        }
      }).catchError((error) {
        print('❌ WebSocket ready error: $error');
        _handleConnectionFailure();
      });

      // Listen to incoming messages
      channelSubscription = channel!.stream.listen(
        (message) {
          if (isPaused || !mounted) return;

          try {
            final data = jsonDecode(message);
            setState(() {
              iScore = (data['algorithm']['i_score'] ?? 0.0).toDouble();
              conditionalComplexity = (data['algorithm']['conditional_complexity'] ?? 0.0).toDouble();
              semiconditionalComplexity = (data['algorithm']['semiconditional_complexity'] ?? 0.0).toDouble();
              iScoreHistory = (data['algorithm']['i_score_history'] as List<dynamic>?)
                  ?.map((e) => (e as num).toDouble())
                  .toList() ?? [];
              
              behaviorVector = (data['visualization']['behavior_vector'] as List<dynamic>?)
                  ?.map((e) => (e as num).toDouble())
                  .toList() ?? [];
              oldPrediction = (data['visualization']['old_prediction'] as List<dynamic>?)
                  ?.map((e) => (e as num).toDouble())
                  .toList() ?? [];
              currentPrediction = (data['visualization']['current_prediction'] as List<dynamic>?)
                  ?.map((e) => (e as num).toDouble())
                  .toList() ?? [];
              
              distanceOldToActual = (data['visualization']['distance_old_to_actual'] ?? 0.0).toDouble();
              distanceCurrentToActual = (data['visualization']['distance_current_to_actual'] ?? 0.0).toDouble();
              distanceDifference = (data['visualization']['distance_difference'] ?? 0.0).toDouble();
              
              if (data['visualization']['trajectory_2d'] != null) {
                trajectory2D = (data['visualization']['trajectory_2d'] as List<dynamic>)
                    .map((point) => (point as List<dynamic>).map((e) => (e as num).toDouble()).toList())
                    .toList();
              }
              
              if (data['visualization']['old_prediction_2d'] != null) {
                oldPrediction2D = (data['visualization']['old_prediction_2d'] as List<dynamic>)
                    .map((e) => (e as num).toDouble())
                    .toList();
              }
              
              if (data['visualization']['current_prediction_2d'] != null) {
                currentPrediction2D = (data['visualization']['current_prediction_2d'] as List<dynamic>)
                    .map((e) => (e as num).toDouble())
                    .toList();
              }
              
              if (data['visualization']['actual_position_2d'] != null) {
                actualPosition2D = (data['visualization']['actual_position_2d'] as List<dynamic>)
                    .map((e) => (e as num).toDouble())
                    .toList();
              }
              
              pcaReady = data['visualization']['pca_ready'] ?? false;
              
              step = data['meta']['step'] ?? 0;
            });
          } catch (e) {
            print('⚠️ Error parsing message: $e');
          }
        },
        onDone: () {
          print('🔌 WebSocket connection closed');
          _handleConnectionFailure();
        },
        onError: (error) {
          print('❌ WebSocket error: $error');
          _handleConnectionFailure();
        },
        cancelOnError: true,
      );
    } catch (e) {
      print('❌ Connection exception: $e');
      _handleConnectionFailure();
    }
  }

  /// Handle connection failure and schedule reconnection
  void _handleConnectionFailure() {
    if (!mounted || !_shouldReconnect) return;

    setState(() {
      isConnected = false;
      _isConnecting = false;
    });

    // Schedule reconnection with exponential backoff
    final delay = _getReconnectDelay();
    print('⏱️  Scheduling reconnection in $delay seconds...');
    
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(Duration(seconds: delay), () {
      if (mounted && _shouldReconnect) {
        connectWebSocket();
      }
    });
  }

  /// Clean up existing connection resources
  void _cleanupConnection() {
    channelSubscription?.cancel();
    channelSubscription = null;
    
    try {
      channel?.sink.close(1000, 'Reconnecting');
    } catch (e) {
      // Ignore close errors
    }
    channel = null;
  }

  void handlePause() {
    setState(() => isPaused = !isPaused);
  }

  void handleReset() {
    print('🔄 Resetting connection...');
    _reconnectAttempts = 0;
    _shouldReconnect = true;
    
    _cleanupConnection();
    _reconnectTimer?.cancel();
    
    setState(() {
      iScore = 0.0;
      iScoreHistory.clear();
      step = 0;
      isConnected = false;
      _isConnecting = false;
    });
    
    Timer(const Duration(milliseconds: 500), connectWebSocket);
  }

  Future<void> handleFreezeLearning() async {
    try {
      final response = await http.post(
        Uri.parse('https://interactivity-agent.onrender.com/experiment/freeze_learning'),
      );
      
      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        
        if (!mounted) return;
        
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Freeze Learning Experiment'),
            content: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Baseline I-Score: ${(result['baseline_i_score'] as num).toStringAsFixed(4)}'),
                  const SizedBox(height: 8),
                  Text('After Freeze: ${(result['average_frozen_i_score'] as num).toStringAsFixed(4)}'),
                  const SizedBox(height: 8),
                  Text('Drop: ${(result['drop'] as num).toStringAsFixed(4)} (${(result['drop_percentage'] as num).toStringAsFixed(1)}%)',
                      style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.red)),
                  const SizedBox(height: 16),
                  const Text('Conclusion:', style: TextStyle(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),
                  Text(result['conclusion'] as String),
                  const SizedBox(height: 16),
                  const Text('This proves Theorem 2: Agent must keep learning to maintain high interactivity!',
                      style: TextStyle(fontStyle: FontStyle.italic)),
                ],
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('OK'),
              ),
            ],
          ),
        );
      } else {
        throw Exception('API returned ${response.statusCode}');
      }
    } catch (e) {
      if (!mounted) return;
      
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $e\n\nMake sure backend is running!'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            const Text('Behavior Space Visualization'),
            const SizedBox(width: 12),
            // Connection status indicator
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: isConnected 
                    ? Colors.green.withOpacity(0.2)
                    : (_isConnecting 
                        ? Colors.orange.withOpacity(0.2)
                        : Colors.red.withOpacity(0.2)),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: isConnected 
                      ? Colors.green
                      : (_isConnecting ? Colors.orange : Colors.red),
                  width: 1,
                ),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: isConnected 
                          ? Colors.green
                          : (_isConnecting ? Colors.orange : Colors.red),
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    isConnected 
                        ? 'Connected'
                        : (_isConnecting 
                            ? 'Connecting...'
                            : 'Disconnected'),
                    style: TextStyle(
                      fontSize: 12,
                      color: isConnected 
                          ? Colors.green.shade700
                          : (_isConnecting 
                              ? Colors.orange.shade700
                              : Colors.red.shade700),
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        backgroundColor: const Color(0xFF667eea),
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _buildMetricsCard(),
                    const SizedBox(height: 16),
                    _buildIScoreHistory(),
                    const SizedBox(height: 16),
                    _buildTrajectoryVisualization(),
                    const SizedBox(height: 16),
                    _buildExplanation(),
                  ],
                ),
              ),
            ),
          ),
          _buildControls(),
          _buildFooter(),
        ],
      ),
    );
  }

  Widget _buildMetricsCard() {
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF667eea), Color(0xFF764ba2)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Live Metrics',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
              ),
              Text(
                'Step: $step',
                style: const TextStyle(fontSize: 14, color: Colors.white70),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(child: _buildMetricBox('I-Score', iScore, '📊', 'Interactivity score measures how much the agent is learning from its environment')),
              const SizedBox(width: 8),
              Expanded(child: _buildMetricBox('Conditional', conditionalComplexity, '🎯', 'Complexity of predicting behavior given history')),
              const SizedBox(width: 8),
              Expanded(child: _buildMetricBox('Semi-Cond.', semiconditionalComplexity, '🔮', 'Complexity of predicting without recent history')),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetricBox(String label, double value, String emoji, String tooltip) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(emoji, style: const TextStyle(fontSize: 20)),
        const SizedBox(height: 4),
        Text(label, style: const TextStyle(fontSize: 10, color: Colors.white70)),
        const SizedBox(height: 4),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.2),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Flexible(
                child: Text(
                  value.toStringAsFixed(4),
                  style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.white),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 2),
        Text(tooltip, style: const TextStyle(fontSize: 9, color: Colors.white60, fontStyle: FontStyle.italic)),
      ],
    );
  }

  Widget _buildIScoreHistory() {
    return Container(
      height: 120,
      padding: const EdgeInsets.all(12.0),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('I-Score Over Time', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Expanded(
            child: iScoreHistory.isEmpty
                ? const Center(child: Text('Collecting data...', style: TextStyle(color: Colors.grey, fontSize: 11)))
                : CustomPaint(
                    size: Size.infinite,
                    painter: LineChartPainter(data: iScoreHistory),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrajectoryVisualization() {
    return Container(
      height: 300,
      padding: const EdgeInsets.all(12.0),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('2D Behavior Trajectory (PCA)', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Expanded(
            child: !pcaReady
                ? const Center(child: Text('Building PCA space... (needs 50+ points)', style: TextStyle(color: Colors.grey, fontSize: 11)))
                : CustomPaint(
                    size: Size.infinite,
                    painter: TrajectoryPainter(
                      trajectory: trajectory2D,
                      oldPrediction: oldPrediction2D,
                      currentPrediction: currentPrediction2D,
                      actualPosition: actualPosition2D,
                    ),
                  ),
          ),
          const SizedBox(height: 8),
          const Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _LegendItem(color: Colors.blue, label: '● Current position'),
              SizedBox(width: 12),
              _LegendItem(color: Colors.orange, label: '× Old prediction'),
              SizedBox(width: 12),
              _LegendItem(color: Colors.green, label: '+ Current prediction'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildExplanation() {
    String explanation;
    if (distanceDifference > 0.01) {
      explanation = 'Old prediction was ${distanceOldToActual.toStringAsFixed(3)} units away, '
          'but current prediction is only ${distanceCurrentToActual.toStringAsFixed(3)} units away. '
          'Memory helped by ${distanceDifference.toStringAsFixed(3)} units! This is strong meta-learning.';
    } else {
      explanation = 'Predictions are similar distance from actual behavior. '
          'Memory is not providing much benefit yet. Agent needs more experience.';
    }

    return Container(
      padding: const EdgeInsets.all(12.0),
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border(left: BorderSide(color: Colors.blue, width: 4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text('💡 What This Means', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text(explanation, style: const TextStyle(fontSize: 12, height: 1.5)),
        ],
      ),
    );
  }

  Widget _buildControls() {
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        alignment: WrapAlignment.center,
        children: [
          ElevatedButton(
            onPressed: handlePause,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blue.shade600,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            ),
            child: Text(isPaused ? '▶️ Resume' : '⏸️ Pause'),
          ),
          ElevatedButton(
            onPressed: handleReset,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.grey.shade600,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            ),
            child: const Text('🔄 Reset'),
          ),
          ElevatedButton(
            onPressed: handleFreezeLearning,
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF764ba2),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            ),
            child: const Text('🧊 Freeze Learning'),
          ),
        ],
      ),
    );
  }

  Widget _buildFooter() {
    return Container(
      color: const Color(0xFF2c3e50),
      padding: const EdgeInsets.all(12.0),
      child: const Text(
        'Pure Technical Implementation: Behavioral Self-Prediction with Geometric I-Score',
        textAlign: TextAlign.center,
        style: TextStyle(color: Colors.white70, fontSize: 11),
      ),
    );
  }

  @override
  void dispose() {
    _shouldReconnect = false;  // Prevent reconnection after dispose
    _reconnectTimer?.cancel();
    _cleanupConnection();
    super.dispose();
  }
}

class _LegendItem extends StatelessWidget {
  final Color color;
  final String label;

  const _LegendItem({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(fontSize: 10)),
      ],
    );
  }
}


class TrajectoryPainter extends CustomPainter {
  final List<List<double>> trajectory;
  final List<double>? oldPrediction;
  final List<double>? currentPrediction;
  final List<double>? actualPosition;

  TrajectoryPainter({
    required this.trajectory,
    this.oldPrediction,
    this.currentPrediction,
    this.actualPosition,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (trajectory.isEmpty) return;

    double minX = trajectory.map((p) => p[0]).reduce(math.min);
    double maxX = trajectory.map((p) => p[0]).reduce(math.max);
    double minY = trajectory.map((p) => p[1]).reduce(math.min);
    double maxY = trajectory.map((p) => p[1]).reduce(math.max);

    final range = math.max(maxX - minX, maxY - minY).clamp(1e-6, double.infinity);
    final centerX = (minX + maxX) / 2;
    final centerY = (minY + maxY) / 2;

    Offset toScreen(List<double> point) {
      final x = (point[0] - centerX) / range * size.width * 0.8 + size.width / 2;
      final y = (point[1] - centerY) / range * size.height * 0.8 + size.height / 2;
      return Offset(x, y);
    }

    final pathPaint = Paint()
      ..color = Colors.blue.withOpacity(0.5)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    final path = Path();
    for (int i = 0; i < trajectory.length; i++) {
      final point = toScreen(trajectory[i]);
      if (i == 0) {
        path.moveTo(point.dx, point.dy);
      } else {
        path.lineTo(point.dx, point.dy);
      }
    }
    canvas.drawPath(path, pathPaint);

    if (oldPrediction != null) {
      _drawPoint(canvas, toScreen(oldPrediction!), Colors.orange, '×', 8);
    }
    if (currentPrediction != null) {
      _drawPoint(canvas, toScreen(currentPrediction!), Colors.green, '+', 8);
    }
    if (actualPosition != null) {
      _drawPoint(canvas, toScreen(actualPosition!), Colors.blue, '●', 10);
    }
  }

  void _drawPoint(Canvas canvas, Offset pos, Color color, String label, double size) {
    final paint = Paint()..color = color;
    canvas.drawCircle(pos, size, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class LineChartPainter extends CustomPainter {
  final List<double> data;

  LineChartPainter({required this.data});

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty || size.width <= 0 || size.height <= 0) return;

    final displayData = data.length > 200 ? data.sublist(data.length - 200) : data;
    final maxValue = displayData.reduce(math.max).clamp(0.1, double.infinity);
    final stepX = size.width / (displayData.length - 1).clamp(1, double.infinity);

    final linePaint = Paint()
      ..color = Colors.blue
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    final path = Path();
    for (int i = 0; i < displayData.length; i++) {
      final x = i * stepX;
      final normalizedValue = (displayData[i] / maxValue).clamp(0.0, 1.0);
      final y = size.height * (1.0 - normalizedValue);

      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    canvas.drawPath(path, linePaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}