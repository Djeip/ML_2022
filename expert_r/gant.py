import datetime
import gantt

# Change font default
gantt.define_font_attributes(fill='black', stroke='black', stroke_width=0, font_family="Verdana")

# Create some tasks
t1 = gantt.Task(name='Сдать зачет по Анализу нечисловой информации', start=datetime.date(2022, 12, 24), duration=1,
                percent_done=100)
t2 = gantt.Task(name='Сдать кт1 по Инструментам управления проектами в условиях риска и неопределенности',
                start=datetime.date(2022, 12, 24), duration=2, percent_done=50)
t3 = gantt.Task(name='Сдать кт2 по Инструментам управления проектами в условиях риска и неопределенности',
                start=datetime.date(2022, 12, 24), duration=2, percent_done=70)
t4 = gantt.Task(name='Сдать кт3 по БД', start=datetime.date(2022, 12, 23), duration=1, percent_done=100)
t5 = gantt.Task(name='Реализовать BERT', start=datetime.date(2022, 12, 23), stop=datetime.date(2022, 12, 31),
                percent_done=0)
t6 = gantt.Task(name='Сдать отчет по Инструментам управления проектами в условиях риска и неопределенности',
                start=datetime.date(2022, 12, 23), depends_of=[t2, t3], duration=20, percent_done=10)
t7 = gantt.Task(name='Завершить практику по НИР и написать отчет', start=datetime.date(2022, 9, 1),
                stop=datetime.date(2022, 12, 29), percent_done=60)
t8 = gantt.Task(name='Cдать преддипломную практику', start=datetime.date(2023, 2, 1), stop=datetime.date(2023, 4, 10),
                percent_done=0)
t9 = gantt.Task(name='Написать и опубликовать статью', start=datetime.date(2022, 9, 1), stop=datetime.date(2023, 6, 1),
                percent_done=0)
t11 = gantt.Task(name='Пересдать алгоритмы оптимизации на графах', start=datetime.date(2023, 2, 1),
                 stop=datetime.date(2023, 6, 1), percent_done=0)
t12 = gantt.Task(name='Написать ВКР', start=datetime.date(2023, 2, 1),
                 stop=datetime.date(2023, 6, 1), percent_done=0)
t13 = gantt.Task(name='Защитить ВКР', start=datetime.date(2023, 6, 1),
                 stop=datetime.date(2023, 7, 1), percent_done=0)
t14 = gantt.Task(name='Сдать экзамен по Эффективным вычислительным алгоритмам',
                 start=datetime.date(2023, 1, 10), duration=1, percent_done=0)
t15 = gantt.Task(name='Сдать экзамен по Инструментам управления проектами в условиях риска и неопределенности',
                 start=datetime.date(2023, 1, 14), duration=1, percent_done=0)

# Create a project
p1 = gantt.Project(name='Дипломирование')

# Add tasks to this project
p1.add_task(t1)
p1.add_task(t2)
p1.add_task(t3)
p1.add_task(t4)
p1.add_task(t5)
p1.add_task(t6)
p1.add_task(t7)
p1.add_task(t8)
p1.add_task(t9)
p1.add_task(t11)
p1.add_task(t12)
p1.add_task(t13)
p1.add_task(t14)
p1.add_task(t15)

##########################$ MAKE DRAW ###############

p1.make_svg_for_tasks(filename='diplomirovanie.svg', today=datetime.date(2022, 12, 24),
                      start=datetime.date(2022, 12, 20),
                      end=datetime.date(2023, 9, 1))

##########################$ /MAKE DRAW ###############
