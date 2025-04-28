import 'package:flutter/material.dart';


class Card extends StatelessWidget {
  const Card({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 100,
      height: 100,
      decoration: BoxDecoration(
        color: Color(0xFF0F9C8F),
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Center(
        child: Text("Text"),
      ),
    );
  }
}

