import 'package:flutter/material.dart';

class Destination {
  const Destination({required this.label, required this.icon});

  final String label;
  final IconData icon;
}

const destinations = [
  Destination(label: 'Главная', icon: Icons.home_outlined),
  Destination(label: 'Соглашение', icon: Icons.book_outlined),
];