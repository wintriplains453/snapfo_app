import 'package:flutter/material.dart';
import '../components/Tabs.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Center(
          child: Text('SNAPFO')
        ),
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          return Column(
            children: [
              Container(
                width: constraints.maxWidth,
                height: constraints.maxHeight * 0.7,
                child: Center(
                  child: Text(
                    '100% Width\n60% Height',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white, fontSize: 20),
                  ),
                ),                
              ),
              Material(
                child: TabsContainer(),
              ),
            ],
          );
        },
      ),
    );
  }
}
