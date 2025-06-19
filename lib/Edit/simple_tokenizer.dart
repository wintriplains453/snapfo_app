import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show DefaultAssetBundle;
import 'dart:convert';
import 'dart:typed_data';
import 'package:archive/archive_io.dart';
import 'package:html/parser.dart' show parse;

class Cache {
  static String? _defaultBpePath;
  static Map<int, String>? _bytesToUnicode;
}

String defaultBpe() {
  return 'assets/bpe_simple_vocab_16e6.txt.gz';
}

Map<int, String> bytesToUnicode() {
  if (Cache._bytesToUnicode != null) {
    return Cache._bytesToUnicode!;
  }

  final bs = <int>[];
  final cs = <int>[];

  // Добавляем символы от ! до ~
  bs.addAll(List.generate(ord('~') - ord('!') + 1, (i) => ord('!') + i));
  // Добавляем символы от ¡ до ¬
  bs.addAll(List.generate(ord('¬') - ord('¡') + 1, (i) => ord('¡') + i));
  // Добавляем символы от ® до ÿ
  bs.addAll(List.generate(ord('ÿ') - ord('®') + 1, (i) => ord('®') + i));

  cs.addAll(bs);

  // Заполняем оставшиеся байты, начиная с кодовой точки 256
  int n = 256; // Начинаем с 256, чтобы не пересекаться с bs
  for (int b = 0; b < 1 << 8; b++) {
    if (!bs.contains(b)) {
      bs.add(b);
      cs.add(n);
      n++;
    }
  }

  final unicodeMap = <int, String>{};
  for (int i = 0; i < bs.length; i++) {
    unicodeMap[bs[i]] = String.fromCharCode(cs[i]);
  }

  Cache._bytesToUnicode = unicodeMap;
  return unicodeMap;
}

// Вспомогательная функция для получения кодовой точки символа
int ord(String char) => char.codeUnitAt(0);

// Аналог get_pairs
Set<(String, String)> getPairs(List<String> word) {
  final pairs = <(String, String)>{};
  if (word.isEmpty) return pairs;

  String prevChar = word[0];
  for (int i = 1; i < word.length; i++) {
    pairs.add((prevChar, word[i]));
    prevChar = word[i];
  }
  return pairs;
}

// Аналог basic_clean (без ftfy, используем html unescape)
String basicClean(String text) {
  // Заменяем ftfy.fix_text на базовую очистку
  // Можно добавить пакет для ftfy, если нужен полный аналог
  text = text.replaceAll(RegExp(r'[^\x20-\x7E]'), ''); // Простая очистка
  text = parse(text).body!.text; // HTML unescape
  return text.trim();
}

// Аналог whitespace_clean
String whitespaceClean(String text) {
  text = text.replaceAll(RegExp(r'\s+'), ' ');
  return text.trim();
}

class SimpleTokenizer {
  late Map<int, String> byteEncoder;
  late Map<String, int> byteDecoder;
  late Map<String, int> encoder;
  late Map<int, String> decoder;
  late Map<(String, String), int> bpeRanks;
  late Map<String, String> cache;
  late RegExp pat;

  SimpleTokenizer._({
    required this.byteEncoder,
    required this.byteDecoder,
    required this.encoder,
    required this.decoder,
    required this.bpeRanks,
    required this.cache,
    required this.pat,
  });

  static Future<SimpleTokenizer> create({String bpePath = '', required BuildContext context}) async {
    bpePath = bpePath.isEmpty ? defaultBpe() : bpePath;

    final byteEncoder = bytesToUnicode();
    print(bytesToUnicode().entries.take(10));
    final byteDecoder = {for (var e in byteEncoder.entries) e.value: e.key};

    // Чтение файла из assets
    final gzipBytes = await DefaultAssetBundle.of(context).load(bpePath);
    final gzipData = gzipBytes.buffer.asUint8List();
    final decompressed = GZipDecoder().decodeBytes(gzipData);
    final merges = utf8.decode(decompressed).split('\n').sublist(1, 49152 - 256 - 2 + 1);

    final mergeTuples = merges.map((m) => m.split(' ')).toList();
    final vocab = byteEncoder.values.toList();
    final vocabWithEnd = vocab.map((v) => '$v</w>').toList();
    vocab.addAll(vocabWithEnd);
    for (var merge in mergeTuples) {
      vocab.add(merge.join(''));
    }
    vocab.addAll(['<|startoftext|>', '<|endoftext|>']);
    print(mergeTuples.length);
    print(mergeTuples.sublist(0, 5));

    final encoder = {for (int i = 0; i < vocab.length; i++) vocab[i]: i};
    final decoder = {for (var e in encoder.entries) e.value: e.key};

    final bpeRanks = {
      for (int i = 0; i < mergeTuples.length; i++)
        (mergeTuples[i][0], mergeTuples[i][1]): i
    };

    final cache = {
      '<|startoftext|>': '<|startoftext|>',
      '<|endoftext|>': '<|endoftext|>'
    };

    final pat = RegExp(
        r'<\|startoftext\|>|<\|endoftext\|>|'
        r"'s|'t|'re|'ve|'m|'ll|'d|"
        r'[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+',
        caseSensitive: false);

    return SimpleTokenizer._(
      byteEncoder: byteEncoder,
      byteDecoder: byteDecoder,
      encoder: encoder,
      decoder: decoder,
      bpeRanks: bpeRanks,
      cache: cache,
      pat: pat,
    );
  }

  String bpe(String token) {
    if (cache.containsKey(token)) {
      return cache[token]!;
    }

    var word = token.split('').toList();
    word = word.sublist(0, word.length - 1) + ['${word.last}</w>'];
    var pairs = getPairs(word);

    if (pairs.isEmpty) {
      return '$token</w>';
    }

    while (true) {
      final bigram = pairs.reduce((a, b) =>
      (bpeRanks[a] ?? double.infinity) < (bpeRanks[b] ?? double.infinity)
          ? a
          : b);
      if (!bpeRanks.containsKey(bigram)) {
        break;
      }

      final (first, second) = bigram;
      final newWord = <String>[];
      int i = 0;
      while (i < word.length) {
        try {
          final j = word.indexOf(first, i);
          newWord.addAll(word.sublist(i, j));
          i = j;
        } catch (e) {
          newWord.addAll(word.sublist(i));
          break;
        }

        if (word[i] == first && i < word.length - 1 && word[i + 1] == second) {
          newWord.add(first + second);
          i += 2;
        } else {
          newWord.add(word[i]);
          i += 1;
        }
      }
      word = newWord;
      if (word.length == 1) {
        break;
      }
      pairs = getPairs(word);
    }

    final result = word.join(' ');
    cache[token] = result;
    return result;
  }

  List<int> encode(String text) {
    final bpeTokens = <int>[];
    text = whitespaceClean(basicClean(text)).toLowerCase();
    final matches = pat.allMatches(text).map((m) => m.group(0)!).toList();

    for (var token in matches) {
      final encodedToken = token.runes.map((r) => byteEncoder[r]!).join('');
      final bpeResult = bpe(encodedToken).split(' ');
      bpeTokens.addAll(bpeResult.map((t) => encoder[t]!));
    }

    return bpeTokens;
  }

  String decode(List<int> tokens) {
    var text = tokens.map((t) => decoder[t]!).join('');
    final bytes = text.runes.map((r) => byteDecoder[String.fromCharCode(r)]!);
    text = utf8.decode(bytes.toList(), allowMalformed: true).replaceAll('</w>', ' ');
    return text;
  }
}

// Аналог функции tokenize
Future<List<List<int>>> tokenize(
    dynamic texts, {
      int contextLength = 77,
      bool truncate = false,
      SimpleTokenizer? tokenizer,
      required BuildContext context,
    }) async {
  tokenizer = tokenizer ?? await SimpleTokenizer.create(context: context);

  if (texts is String) {
    texts = [texts];
  }

  if (tokenizer == null) {
    throw Exception('Tokenizer cannot be null');
  }

  final sotToken = tokenizer.encoder['<|startoftext|>']!;
  final eotToken = tokenizer.encoder['<|endoftext|>']!;
  final allTokens = texts.map((t) => [sotToken, ...tokenizer!.encode(t), eotToken]).toList();

  final result = List.generate(
      allTokens.length, (_) => List.filled(contextLength, 0, growable: false));

  for (int i = 0; i < allTokens.length; i++) {
    var tokensList = allTokens[i];
    if (tokensList.length > contextLength) {
      if (truncate) {
        tokensList = tokensList.sublist(0, contextLength);
        tokensList[contextLength - 1] = eotToken;
      } else {
        throw Exception(
            'Input ${texts[i]} is too long for context length $contextLength');
      }
    }
    for (int j = 0; j < tokensList.length; j++) {
      result[i][j] = tokensList[j];
    }
  }

  return result;
}

void main() {
}