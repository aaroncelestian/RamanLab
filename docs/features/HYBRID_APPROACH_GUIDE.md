# Как использовать гибридный подход для планирования

Этот документ объясняет, как создан план реализации панели результатов и как работать с ним дальше.

## 📁 Что было создано

### 1. Подробный план реализации
**Файл:** `docs/features/results-panel-implementation-plan.md`

Полный технический документ, который включает:
- Описание функциональности
- Архитектурные решения
- 7 фаз реализации с детальными задачами
- Структуру файлов
- Примеры кода
- Оценки времени (15-20 часов)
- Критерии успеха

**Преимущества:**
- Вся информация в одном месте
- Версионируется вместе с кодом
- Можно обновлять по мере работы
- Служит документацией на будущее

### 2. Индекс документации
**Файл:** `docs/features/README.md`

Оглавление всех feature планов в репозитории:
- Список активных планов
- Статусы и приоритеты
- Инструкции по использованию

### 3. Шаблон для GitHub Issue
**Файл:** `docs/features/ISSUE_TEMPLATE_results_panel.md`

Готовый текст для создания epic issue в GitHub:
- Краткое описание функций
- Список всех фаз с чекбоксами
- Ссылка на полный план
- Критерии успеха

## 🚀 Следующие шаги

### Шаг 1: Создать Epic Issue в GitHub

1. Зайти на https://github.com/timnik82/RamanLab/issues/new
2. Скопировать содержимое файла `docs/features/ISSUE_TEMPLATE_results_panel.md`
3. Вставить как текст issue
4. Добавить labels: `feature`, `enhancement`, `peak-fitting`, `ui`
5. Создать issue

**Результат:** Получите issue с номером (например, #7), который будет главным tracking issue

### Шаг 2: Начать реализацию

#### Вариант А: Реализовывать последовательно

Для каждой фазы:
1. Создать отдельный issue (опционально):
   - Title: "Phase 1: Base Panel Structure"
   - Ссылка на epic: "Part of #7"
   - Описание из плана
2. Создать ветку: `git checkout -b feature/results-panel-phase1`
3. Реализовать задачи из фазы
4. Коммиты: `git commit -m "Implement base panel structure. Refs #7"`
5. Push и создать PR с пометкой "Part of #7"

#### Вариант Б: Реализовывать всё сразу

1. Создать одну ветку: `git checkout -b feature/results-panel`
2. Реализовывать все фазы последовательно
3. Коммиты: `git commit -m "Phase 1: Base panel. Refs #7"`
4. Один большой PR в конце

**Рекомендация:** Вариант А (по фазам) лучше для:
- Получения промежуточной обратной связи
- Упрощения code review
- Возможности использовать частичный функционал

### Шаг 3: Отслеживание прогресса

**В Epic Issue (#7):**
- Отмечать галочками завершенные задачи
- Добавлять номера sub-issues по мере создания
- Обновлять, если планы меняются

**В коммитах:**
```bash
# Для промежуточных коммитов
git commit -m "Add statistics computation. Refs #7"

# Для завершения фазы/задачи
git commit -m "Complete Phase 2: Overall statistics display. Refs #7"

# Для завершения всего feature (закрывает issue)
git commit -m "Complete Results Panel implementation. Closes #7"
```

**В pull requests:**
- Title: "Feature: Results Panel - Phase 1" или "Feature: Results Panel (complete)"
- Description: ссылка на epic "Closes #7" или "Part of #7"

### Шаг 4: Обновление документации

По мере реализации:
1. Если архитектура меняется → обновить план
2. Если добавляются новые фазы → добавить в план
3. Если что-то отменяется → пометить в плане
4. Коммит изменений плана вместе с кодом

## 📊 Структура работы

```
Epic Issue #7 (GitHub)
  │
  ├─ Фаза 1 → Issue #8 (опц.)
  │   ├─ Commit: "Create base panel. Refs #7"
  │   ├─ Commit: "Add to main window. Refs #7"
  │   └─ PR #9 → "Part of #7"
  │
  ├─ Фаза 2 → Issue #10 (опц.)
  │   ├─ Commit: "Add statistics computation. Refs #7"
  │   └─ PR #11 → "Part of #7"
  │
  └─ ... (остальные фазы)
```

## 💡 Советы по реализации

### Начать с MVP (Minimum Viable Product)
Фазы 1-3 (8-10 часов) дают работающую базовую функциональность:
- Панель появляется после фиттинга
- Показывает общую статистику
- Клик на карте показывает детали

Это позволит:
- ✅ Быстро получить feedback от пользователей
- ✅ Проверить, что архитектура правильная
- ✅ Начать использовать feature раньше

Фазы 4-7 добавляются потом по необходимости.

### Использовать существующий код
В плане указаны функции для переиспользования:
- `compute_integrated_intensity()` из `core/math_models.py`
- `ConfigManager` для сохранения настроек
- Существующие Qt widgets и стили

### Тестировать на реальных данных
После каждой фазы:
1. Запустить приложение
2. Загрузить карту кремния
3. Сделать peak fitting
4. Проверить, что новая функциональность работает

## 📝 Пример workflow

```bash
# 1. Создать Epic Issue в GitHub (получить номер, например #7)

# 2. Создать ветку для Phase 1
git checkout -b feature/results-panel-phase1

# 3. Создать файлы для Phase 1
# map_analysis_2d/ui/results_panel.py
# ... код ...

# 4. Коммит
git add map_analysis_2d/ui/results_panel.py
git commit -m "Create ResultsPanel base class

Implement QDockWidget-based panel with two sections:
- OverallStatsWidget for aggregate statistics
- PixelDetailsWidget for point-specific data

Panel can be docked, floated, and hidden via View menu.

Phase 1 of Results Panel implementation.
Refs #7"

# 5. Интеграция в main_window.py
git add map_analysis_2d/ui/main_window.py
git commit -m "Integrate ResultsPanel into main window

Add results panel as dockable widget on right side.
Add View menu option to show/hide panel.

Refs #7"

# 6. Push и PR
git push -u origin feature/results-panel-phase1
gh pr create --title "Feature: Results Panel - Phase 1" --body "Part of #7

Implements Phase 1: Base Panel Structure
- Creates ResultsPanel as QDockWidget
- Adds to main window
- View menu integration

See implementation plan: docs/features/results-panel-implementation-plan.md"

# 7. После merge, перейти к Phase 2
git checkout main
git pull
git checkout -b feature/results-panel-phase2
# ... повторить процесс ...
```

## 🎯 Цель гибридного подхода

**Документация в репозитории + Tracking в GitHub = Лучшее из обоих подходов**

### Преимущества:
✅ **Подробный план** всегда доступен и версионирован
✅ **GitHub Issues** для tracking и обсуждений
✅ **Гибкость** в разделении работы на части
✅ **Документация** остается после завершения
✅ **История** решений и изменений

### Избегаем проблем:
❌ Фрагментированная информация (все issues разрозненные)
❌ Потерянная документация (только issues, которые закрываются)
❌ Негибкость (монолитный mega-issue)
❌ Отсутствие контекста (коммиты без ссылок на issue)

## 📞 Когда обновлять план

План нужно обновлять, если:
- Меняется архитектура или технический подход
- Добавляются новые требования
- Обнаруживаются неучтенные сложности
- Меняется приоритизация фаз
- Появляется важная информация (performance issues, и т.д.)

План НЕ нужно обновлять для:
- Мелких багфиксов
- Изменений в реализации без изменения подхода
- Рефакторинга без изменения функциональности

---

**Готово!** Теперь можно начинать реализацию. Удачи! 🚀
