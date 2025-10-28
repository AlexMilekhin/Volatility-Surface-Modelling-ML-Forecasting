{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNDKdhiWnCM8fcCwv966JuF",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/AlexMilekhin/Volatility-Surface-Modelling-ML-Forecasting/blob/main/README.md\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Volatility Surface Modelling & ML Forecasting\n",
        "\n",
        "This project builds an end-to-end **options analytics system** using Python and real market data from Yahoo Finance.  \n",
        "It visualises implied volatility surfaces, fits SVI parameters, and applies **machine learning** to forecast volatility dynamics.\n",
        "\n",
        "## 📁 Structure\n",
        "| Notebook | Description |\n",
        "|-----------|--------------|\n",
        "| `1_data_collection.ipynb` | Collect option-chain data |\n",
        "| `2_IV_compute.ipynb` | Clean data and compute IV's |\n",
        "| `3_IV_Analysis_SVI.ipynb` | Compute IVs, features, fit SVI, plot surfaces |\n",
        "| `4_ml_forecasting.ipynb` | Predict next-day ATM IV change using ML |\n",
        "| `utils/helpers.py` | Shared functions for BS pricing and plotting |\n",
        "\n",
        "## 📊 Highlights\n",
        "- Dynamic SVI calibration\n",
        "- Volatility term-structure analysis\n",
        "- Real-data and synthetic surface generation\n",
        "- Polished visual outputs (Matplotlib/Plotly)\n",
        "\n",
        "## 🧠 Skills Demonstrated\n",
        "Python · Quantitative Finance · Machine Learning · Data Engineering · Options Modelling\n"
      ],
      "metadata": {
        "id": "p_xskEsLLNWn"
      }
    }
  ]
}