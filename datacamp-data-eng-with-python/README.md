# DataCamp: Data Engineering with Python

## Overview

In any data-driven company, you will undoubtedly cross paths with data engineers. Among other things, they facilitate some of your work by making data readily available to everyone within the organization, and possibly in bringing machine learning models into production. One way to speed up this process is through building an understanding of what it means to bring processes into production and what features are of high-grade code. In this course, we’ll be looking at various data pipelines the data engineer is building, and how some of the tools he or she is using can help you in getting your models into production or run repetitive tasks consistently and efficiently.

In this course, we illustrate common elements of data engineering pipelines. In Chapter 1, you will learn how to ingest data. Chapter 2 will go one step further with cleaning and transforming data. In Chapter 3, you will learn how to safely deploy code. Finally, in Chapter 4 you will schedule complex dependencies between applications.

Building Data Engineering Pipelines covers new technologies and material, so we recommend that you have a strong understanding of the prerequisites to get the most out of this course.

## Section 1: Ingesting Data

After seeing this chapter, you will be able to explain what a data platform is, how data ends up in it, and how data engineers structure its foundations. You will be able to ingest data from a RESTful API into the data platform’s data lake using a self-written ingestion pipeline, made using Singer’s taps and targets.

https://campus.datacamp.com/courses/building-data-engineering-pipelines-in-python

## Section 2: Creating a data transformation pipeline with PySpark

You will learn how to process data in the data lake in a structured way using PySpark. Of course, you must first understand when PySpark is the right choice for the job.

## Section 3: Testing your data pipeline

Stating “it works on my machine” is not a guarantee it will work reliably elsewhere and in the future. Requirements for your project will change. In this chapter, we explore different forms of testing and learn how to write unit tests for our PySpark data transformation pipeline, so that we make robust and reusable parts.

## Section 4: Managing and orchestrating a workflow

We will explore the basics of Apache Airflow, a popular piece of software that allows you to trigger the various components of an ETL pipeline on a certain time schedule and execute tasks in a specific order. Here too, we illustrate how a deployment of Apache Airflow can be tested automatically.