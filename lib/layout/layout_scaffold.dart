import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../router/destination.dart';

class LayoutScaffold extends StatelessWidget {
  const LayoutScaffold({
    required this.navigationShell,
    Key? key,
  }) : super(key: key ?? const ValueKey<String>('LayoutScaffold'));

  final StatefulNavigationShell navigationShell;

  @override
  Widget build(BuildContext context) => Scaffold(
    body: navigationShell,
    bottomNavigationBar: SizedBox(
      height: 52.0, // Установите желаемую высоту
      child: BottomNavigationBar(
        showSelectedLabels: false,
        showUnselectedLabels: false,
        iconSize: 18,
        currentIndex: navigationShell.currentIndex,
        onTap: navigationShell.goBranch,
        backgroundColor: Theme.of(context).primaryColor,
        selectedItemColor: Colors.white,
        unselectedItemColor: Colors.grey,
        items: destinations
            .map((destination) => BottomNavigationBarItem(
          icon: Icon(destination.icon),
          label: destination.label,
          activeIcon: Icon(destination.icon, color: Colors.white),
        ))
            .toList(),
      ),
    ),
  );
}