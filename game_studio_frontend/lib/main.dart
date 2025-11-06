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
  late WebSocketChannel channel;
  bool isConnected = false;
  bool isPaused = false;

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
    connectWebSocket();
  }

  void connectWebSocket() {
    try {
      channel = WebSocketChannel.connect(
        Uri.parse('ws://localhost:8000/ws/agent'),
      );

      channel.ready.then((_) {
        if (mounted) {
          setState(() => isConnected = true);
          print('Connected to behavior space agent');
        }
      });

      channel.stream.listen(
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
            print('Error parsing message: $e');
          }
        },
        onDone: () {
          if (mounted) {
            setState(() => isConnected = false);
            print('WebSocket closed');
          }
        },
        onError: (error) {
          if (mounted) {
            setState(() => isConnected = false);
            print('WebSocket error: $error');
          }
        },
      );
    } catch (e) {
      if (mounted) {
        setState(() => isConnected = false);
        print('Connection failed: $e');
      }
    }
  }

  void handlePause() {
    setState(() => isPaused = !isPaused);
  }

  void handleReset() {
    channel.sink.close(1000, 'Resetting');
    setState(() {
      iScore = 0.0;
      iScoreHistory.clear();
      step = 0;
      isConnected = false;
    });
    Timer(const Duration(milliseconds: 100), connectWebSocket);
  }

  Future<void> handleFreezeLearning() async {
    try {
      final response = await http.post(
        Uri.parse('http://localhost:8000/experiment/freeze_learning'),
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
        title: const Text('Behavior Space Visualization'),
        backgroundColor: const Color(0xFF667eea),
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          _buildHeader(),
          Expanded(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final isWide = constraints.maxWidth > 1000;
                return SingleChildScrollView(
                  padding: const EdgeInsets.all(16.0),
                  child: isWide
                      ? Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Expanded(child: _buildLeftPanel()),
                            const SizedBox(width: 16),
                            Expanded(child: _buildRightPanel()),
                          ],
                        )
                      : Column(
                          children: [
                            _buildLeftPanel(),
                            const SizedBox(height: 16),
                            _buildRightPanel(),
                          ],
                        ),
                );
              },
            ),
          ),
          _buildControls(),
          _buildFooter(),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      color: const Color(0xFF764ba2),
      padding: const EdgeInsets.symmetric(vertical: 12.0),
      child: Column(
        children: [
          const Text(
            'Pure Technical Visualization: 64-Dimensional Behavior Space',
            style: TextStyle(color: Colors.white70, fontSize: 14),
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: isConnected ? Colors.green.shade700 : Colors.red.shade700,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  isConnected ? '🟢 Connected' : '🔴 Disconnected',
                  style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                ),
              ),
              const SizedBox(width: 16),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.white24,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  'Step: $step',
                  style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildLeftPanel() {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '📊 Behavior Vectors (64-Dimensional)',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'Raw behavior space visualization. No abstractions.',
              style: TextStyle(color: Colors.grey, fontSize: 12),
            ),
            const SizedBox(height: 16),
            
            // Behavior vector bars
            _buildVectorBars(),
            
            const SizedBox(height: 24),
            
            // 2D trajectory
            if (pcaReady) ...[
              const Text(
                'Trajectory (PCA: 64D → 2D)',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              _buildTrajectoryPlot(),
            ] else ...[
              const SizedBox(
                height: 200,
                child: Center(
                  child: Text('Collecting data for PCA projection...',
                      style: TextStyle(color: Colors.grey)),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildRightPanel() {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Geometric I-Score Interpretation',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'I-Score = Distance between predictions (geometric complexity)',
              style: TextStyle(color: Colors.grey, fontSize: 12),
            ),
            const SizedBox(height: 16),
            
            // I-Score card
            _buildIScoreCard(),
            
            const SizedBox(height: 16),
            
            // Distance breakdown
            _buildDistanceBreakdown(),
            
            const SizedBox(height: 16),
            
            // I-Score history
            _buildIScoreHistory(),
            
            const SizedBox(height: 16),
            
            // Explanation
            _buildExplanation(),
          ],
        ),
      ),
    );
  }

  Widget _buildVectorBars() {
    if (behaviorVector.isEmpty) {
      return const SizedBox(
        height: 100,
        child: Center(child: CircularProgressIndicator()),
      );
    }

    // Show first 32 dimensions
    final displayVector = behaviorVector.take(32).toList();

    return SizedBox(
      height: 100,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: List.generate(displayVector.length, (i) {
          final value = displayVector[i];
          final barHeight = (value.abs() * 80).clamp(5.0, 80.0);
          final color = value > 0 ? Colors.blue : Colors.red;

          return Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 0.5),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.end,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    height: barHeight,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: color,
                      borderRadius: const BorderRadius.vertical(top: Radius.circular(2)),
                    ),
                  ),
                  const SizedBox(height: 2),
                  if (i % 4 == 0)
                    Text(
                      '$i',
                      style: const TextStyle(fontSize: 8, color: Colors.grey),
                    ),
                ],
              ),
            ),
          );
        }),
      ),
    );
  }

  Widget _buildTrajectoryPlot() {
    if (trajectory2D.isEmpty) {
      return const SizedBox(
        height: 200,
        child: Center(child: Text('No trajectory data')),
      );
    }

    return SizedBox(
      height: 250,
      child: CustomPaint(
        size: Size.infinite,
        painter: TrajectoryPainter(
          trajectory: trajectory2D,
          oldPrediction: oldPrediction2D,
          currentPrediction: currentPrediction2D,
          actualPosition: actualPosition2D,
        ),
      ),
    );
  }

  Widget _buildIScoreCard() {
    String interpretation;
    if (iScore > 0.05) {
      interpretation = '🟢 Strong Meta-Learning';
    } else if (iScore > 0.01) {
      interpretation = '🟡 Moderate Meta-Learning';
    } else {
      interpretation = '🔴 Weak Meta-Learning';
    }

    return Container(
      padding: const EdgeInsets.all(20.0),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        gradient: const LinearGradient(
          colors: [Color(0xFF667eea), Color(0xFF764ba2)],
        ),
      ),
      child: Column(
        children: [
          const Text(
            'CURRENT I-SCORE',
            style: TextStyle(fontSize: 12, color: Colors.white70, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          Text(
            iScore.toStringAsFixed(4),
            style: const TextStyle(fontSize: 36, color: Colors.white, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(
            interpretation,
            style: const TextStyle(fontSize: 14, color: Colors.white, fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }

  Widget _buildDistanceBreakdown() {
    return Container(
      padding: const EdgeInsets.all(12.0),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            'Distance Analysis',
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),
          _buildDistanceBar(
            'Old Prediction → Actual',
            distanceOldToActual,
            Colors.orange,
            'How far was prediction from H steps ago',
          ),
          const SizedBox(height: 8),
          _buildDistanceBar(
            'Current Prediction → Actual',
            distanceCurrentToActual,
            Colors.green,
            'How far is current prediction',
          ),
          const SizedBox(height: 8),
          const Divider(),
          const SizedBox(height: 8),
          _buildDistanceBar(
            'Difference (Geometric I-Score)',
            distanceDifference,
            Colors.blue,
            'How much memory helped',
            isDifference: true,
          ),
        ],
      ),
    );
  }

  Widget _buildDistanceBar(String label, double value, Color color, String tooltip,
      {bool isDifference = false}) {
    final barWidth = ((value.abs() * 50).clamp(0.0, 100.0));

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(label, style: const TextStyle(fontSize: 11, color: Colors.grey, fontWeight: FontWeight.w600)),
        const SizedBox(height: 4),
        Row(
          children: [
            Expanded(
              child: Container(
                height: 20,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: FractionallySizedBox(
                  widthFactor: barWidth / 100.0,
                  alignment: Alignment.centerLeft,
                  child: Container(
                    decoration: BoxDecoration(
                      color: color,
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            SizedBox(
              width: 60,
              child: Text(
                (isDifference ? '+' : '') + value.toStringAsFixed(4),
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: isDifference ? FontWeight.bold : FontWeight.normal,
                  color: isDifference ? color : Colors.black87,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 2),
        Text(tooltip, style: const TextStyle(fontSize: 9, color: Colors.grey, fontStyle: FontStyle.italic)),
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
    channel.sink.close();
    super.dispose();
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