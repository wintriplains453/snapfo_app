import 'package:flutter/material.dart';

class TabsContainer extends StatelessWidget {
  const TabsContainer({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: MediaQuery.of(context).size.height * 0.24, // 40% высоты экрана
      child: DefaultTabController(
        length: 4,
        child: Column(
          children: [
            TabBar(
              tabs: [
                Tab(icon: Icon(Icons.home)),
                Tab(icon: Icon(Icons.favorite)),
                Tab(icon: Icon(Icons.settings)),
                Tab(icon: Icon(Icons.person)),
              ],
            ),
            Expanded(
              child: TabBarView(
                children: [
                  Container(color: Colors.orange),
                  Container(color: Colors.blue),
                  Container(color: Colors.green),
                  Container(color: Colors.red),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
