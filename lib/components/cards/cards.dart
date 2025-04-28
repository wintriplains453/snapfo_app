import 'package:flutter/material.dart';
import 'package:snapfo_app/components/cards/card/card.dart' as custom;

class Cards extends StatefulWidget {
  const Cards({super.key});

  @override
  State<Cards> createState() => _CardsState();
}

class _CardsState extends State<Cards> {
  @override
  Widget build(BuildContext context) {
    return Container(
      color: Color(0xFF191919),
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: 10,
        itemBuilder: (context, index) {
          return const Padding(
            padding: EdgeInsets.all(8.0),
            child: SizedBox(
              child: custom.Card(),
            ),
          );
        },
      ),
    );
  }
}
