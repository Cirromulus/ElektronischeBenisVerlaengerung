\ProvidesClass{mat}
\LoadClass[11pt,a4paper,oneside]{scrartcl}

\usepackage[utf8]{inputenc}
\usepackage[ngerman]{babel}
\usepackage{amsmath,amssymb}
\usepackage{graphics,graphicx}
\usepackage[T1]{fontenc}
\usepackage{fancyhdr}
\usepackage{multicol}
\usepackage{parskip}
\usepackage{graphicx}
\usepackage{wasysym}
\usepackage{stmaryrd}
\usepackage{listings}
\usepackage{tikz}
\usetikzlibrary{automata,positioning}

\newcommand{\thissemester}{SoSe 17}
\newcommand{\thiscourse}{Elektronische Bildverarbeitung}
\newcommand{\thisdozent}{Udo Frese}
\newcommand{\thistutor}{Gruppe ??}
\newcommand{\thiscourseshort}{T}

\pagestyle{fancy}
\thispagestyle{empty}

% -----------------------------------------------------------------------
% Main macros

\renewcommand{\maketitle}[8]
          {\@maketitle{#1}{#2}{#3}{#4}{#5}{#6}{#7}{#8}}
%\renewcommand\thesection{\alph{section}}

\renewcommand{\@maketitle}[9]
{\begin{tabular}{p{7cm}|r}

\begin{tabular}{l}
{#1} \textit{\scriptsize{{#2}}} \\
{#3} \textit{\scriptsize{{#4}}} \\ 
{#5} \textit{\scriptsize{{#6}}} 
\end{tabular}
&
\begin{tabular}{ll}
Dozent: & \thisdozent \\
Gruppe: & \thistutor \\
Abgabedatum: &  {#7}
\end{tabular}
\end{tabular}

\vspace{15pt}

\Large{\textbf{\thiscourse} (\thissemester)}

 \rule{\linewidth}{1pt}

\Large{\textit{L\"osungsvorschlag zu Übungsblatt {#8}:}}
\\[10pt]
\normalsize
\noindent
    \fancyhead{} %clear all fields
    \fancyhead[LO,RE]{\thiscourseshort, \thissemester}
    \fancyhead[RO,LE]{Gruppe: \textit{{#1} et al}}
}

%qed zeichen
\newcommand{\qed}{\begin{flushright}$^\blacksquare$\end{flushright}}

\newcommand{\geg}{\textbf{Gegeben: }}

\newcommand{\ges}{\textbf{Gesucht: }}

\newcommand{\beh}{\textbf{Behauptung: }}

\newcommand{\bew}{\textbf{Beweis: }}

\endinput
