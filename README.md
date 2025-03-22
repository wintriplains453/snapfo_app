# Изменение и вывод изображения 

Для изменения изображения перед сохранением использовалась библиотека OpenCV
- применения фильтра черно-белого
- применения фильтра размытия

## Краткая сводка

в файле main.dart

основная функция вывода измененного изображения - Image.memory
функция ввода при выборе изображения - picker.pickImage
функция ввода по умолчанию - DefaultAssetBundle.of(context).load

## Принципы работы

при нажатии на кнопку "по умолчанию" происходит превращение файла в байты

```typescript
final data = await DefaultAssetBundle.of(context).load("assets/images/smith_aligned.jpg");
final bytes = data.buffer.asUint8List();
```

в другом сценарии выбора своей картинки преобразуется в объект cv.Mat

```typescript
final mat = cv.imread(path);
```

в состояние добавляется массив изменения для отображения

```typescript
  setState(() {
    images = [bytes, cv.imencode(".png", gray).$2, cv.imencode(".png", blur).$2];
  });
```

или 

```typescript
  setState(() {
    images = [
      cv.imencode(".png", mat).$2,
      cv.imencode(".png", gray).$2,
      cv.imencode(".png", blur).$2,
    ];
  });
```

в виджет прокрутки добавляется длина массива и происходит перебор массива images из состояния выводя по индексу idx images[idx] изображения

```typescript
  Expanded(
    child: ListView.builder(
      itemCount: images.length,
      itemBuilder: (ctx, idx) => Card(
        child: Image.memory(images[idx]),
      ),
    ),
  ),
```
