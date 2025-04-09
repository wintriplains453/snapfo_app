import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:snapfo_app/router/routes.dart';
import 'package:snapfo_app/screens/home_page.dart';
import '../layout/layout_scaffold.dart';
import '../screens/user_agreement.dart';

final _rootNavigatorKey = GlobalKey<NavigatorState>(debugLabel: 'root');


final router = GoRouter(
  navigatorKey: _rootNavigatorKey,
  initialLocation: Routes.homepage,
  routes: [
    StatefulShellRoute.indexedStack(
      builder: (context, state, navigationShell) => LayoutScaffold(
        navigationShell: navigationShell,
      ),
      branches: [
        StatefulShellBranch(
          routes: [
            GoRoute(
              path: Routes.homepage,
              builder: (context, state) => const HomePage(),
            ),
          ],
        ),
        StatefulShellBranch(
          routes: [
            GoRoute(
              path: Routes.useragreement,
              builder: (context, state) => const UserAgreement(),
            ),
          ],
        ),
      ],
    ),
  ],
);