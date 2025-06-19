import 'package:flutter/material.dart';

const primaryColor = Color(0x008F9C0F);

final darkTheme = ThemeData(
  useMaterial3: true,
  primaryColor: primaryColor,
  textTheme: textTheme,
  appBarTheme: appBarTheme,
  scaffoldBackgroundColor: Color.fromARGB(25,25,25,0),
  colorScheme: ColorScheme.fromSeed(
    seedColor: primaryColor,
    brightness: Brightness.dark,
  )
);

const textTheme = TextTheme(
  titleMedium: TextStyle(
    fontSize: 16,
    fontWeight: FontWeight.w600,
  ),
  headlineLarge: TextStyle(
    fontSize: 28,
    fontWeight: FontWeight.w600,
  )
);

const appBarTheme = AppBarTheme(
  backgroundColor: Color.fromARGB(35,35,35,0),
  titleTextStyle: TextStyle(
    color: Colors.white,
    fontWeight: FontWeight.w400,
    fontSize: 20,
  ),
);