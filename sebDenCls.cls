% initial 2009/10/27 Ingo Sieverdingbeck
% ti2.cls adapted by Sebastian Hellmann 2013
% modified for mathmatical proof by Dennis Hemker 2014

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sebCls}[2014/04/28 Sebastian Hellmann, Dennis Hemker]
\LoadClass[a4paper]{scrartcl}

% -----------------------------------------------------------------------
% Required packages

\RequirePackage[ngerman]{babel}
\RequirePackage{amssymb}
\RequirePackage{epsfig}
\RequirePackage{graphicx}
\RequirePackage{moreverb}
\RequirePackage{parskip}
\RequirePackage[utf8]{inputenc} 
\RequirePackage{ifthen} 

% -----------------------------------------------------------------------
% define page layout using typearea

\KOMAoptions{headinclude=true}
\KOMAoptions{footinclude=true}
\areaset{16cm}{27cm}
% -----------------------------------------------------------------------
% define page style
\pagestyle{plain}
\thispagestyle{plain}

% -----------------------------------------------------------------------
% Macro für das Abgabeblatt

\newcommand{\blatt}[5]
{
\begin{tabular}[t]{lr}
  \begin{tabular}[t]{l}
    \fa\hspace{\abstand} \se\\ Dozent/Tutor: {#1}\\ \hspace{70ex} \\
    {\Huge\textsf{{\"Ubungsblatt} {\blattnr}}} %\hspace{10ex} \begin{tabular}{c|c|c|c|c|c}1&2&3&4&5&$\sum$\\\hline &&&&&\end{tabular}
    % Bewertungskästchen
    \\[1.5ex]
    {\Large L\"osungsvorschlag} \\[1ex]
    \ab
    
  \end{tabular}
  &
  \begin{tabular}[t]{l}
      \tut
      {#2}\\
      {#3}\\
      {#4}\\
      {#5}\\
  \end{tabular} \\[2ex]
\end{tabular}\\
\rule{\linewidth}{1pt}
}

\newcommand{\grenzop}[3]{\mathop{#1}\limits^{#2}_{#3}}

\newcommand{\langrechtspfeil}[2]{\grenzop{\longrightarrow}{#1}{#2}}
\newcommand{\bob}[0]{\hspace*{\fill}$\Box$}
% -----------------------------------------------------------------------



%%% Local Variables: 
%%% mode: latex
%%% TeX-master: t
%%% End: 

\usepackage{listings}
\usepackage[usenames,dvipsnames]{color}

\lstset{ %
  backgroundcolor=\color{white},   % choose the background color; you must add \usepackage{color} or \usepackage{xcolor}
  basicstyle=\footnotesize,        % the size of the fonts that are used for the code
  breakatwhitespace=false,         % sets if automatic breaks should only happen at whitespace
  breaklines=true,                 % sets automatic line breaking
  captionpos=b,                    % sets the caption-position to bottom
  commentstyle=\color{Gray},    % comment style
  extendedchars=true,              % lets you use non-ASCII characters; for 8-bits encodings only, does not work with UTF-8 
  keepspaces=true,                 % keeps spaces in text, useful for keeping indentation of code (possibly needs columns=flexible)
  keywordstyle=\color{blue},       % keyword style
  language=C++,                 % the language of the code
  numbers=left,                    % where to put the line-numbers; possible values are (none, left, right)
  numberstyle=\tiny, % the style that is used for the line-numbers
  rulecolor=\color{black},         % if not set, the frame-color may be changed on line-breaks within not-black text (e.g. comments (green here))
  showspaces=false,                % show spaces everywhere adding particular underscores; it overrides 'showstringspaces'
  showstringspaces=false,          % underline spaces within strings only
  showtabs=false,                  % show tabs within strings adding particular underscores
  tabsize=4,                       % sets default tabsize to 2 spaces
  title=\lstname                   % show the filename of files included with \lstinputlisting; also try caption instead of title
}
\lstset{literate=%
{Ã–}{{\"O}}1
{Ã„}{{\"A}}1
{Ãœ}{{\"U}}1
{ÃŸ}{{\ss}}1
{Ã¼}{{\"u}}1
{Ã¤}{{\"a}}1
{Ã¶}{{\"o}}1
{Â°}{{$^\circ$}}1
}

%---------------------------------------------------------------------------

%\RequirePackage{intcalc}
%\RequirePackage{datenumber}

%\setdatetoday 

%Berechnet die Nummer des aktuellen Ãœbungsblattes aus dem Datum
%Für die Blattnummer wird ein Counter erstellt, der bei dem Befehl \blattnr ausgelesen wird.
%\newcounter{blattnr}
%\setcounter{blattnr}{\intcalcDiv{\arabic{datenumber}-78108}{7}}

% Es werden drei Counter erstellt, die mit dem Kommando darauf gesetzt werden. Sie stehen für das Abgabedatum.
%\newcounter{abday}
%\newcounter{abmonth}
%\newcounter{abyear}

%\newcommand{\blattnr}{\arabic{blattnr}}
%\setmydatebynumber{\intcalcAdd{78114}{\intcalcMul{\blattnr}{7}}}{abyear}{abmonth}{abday}

%\newcommand{\abgabe}{\arabic{abday}.\arabic{abmonth}.\arabic{abyear}}


\renewcommand\thesection{Aufgabe \arabic{section}}

\def\ccode{\@ifstar{\@bccode}{\@accode}}
\def\@accode[#1,#2,#3]{\lstinputlisting[linerange={#2-#3},firstnumber=#2]{#1}}
\def\@bccode[#1]{\lstinputlisting{#1}}

\usepackage{enumitem}
\usepackage{amsmath}
\usepackage{wasysym}

%----added sose 2014--------------------------
\newcommand{\abstand}{20ex}

\newcommand{\tut}{}
\newcommand{\tutnr}[1]{\renewcommand{\tut}{#1\\\\}}

\newcommand{\blattnr}{}
\newcommand{\nr}[1]{\renewcommand{\blattnr}{#1}}

\newcommand{\ab}{}
\newcommand{\abgabe}[1]{\renewcommand{\ab}{{\Large Abgabe: {#1}} \\[1ex]}}

\newcommand{\fa}{Informationssicherheit}
\newcommand{\fach}[1]{\renewcommand{\fa}{#1}}

\newcommand{\se}{SoSe 2015}
\newcommand{\semester}[1]{\renewcommand{\se}{#1}}



%---added for mathmatical proof----------------------
\newcommand{\proof}[5]{
	\ifx.#1
		\ifx.#5 & & #2 \overset{#3}{=}& \ #4 & & \\
		\else & & #2 \overset{#3}{=}& \ #4 & & |\ #5 \\ \fi
	\else 
		\ifx.#5\stackrel{#1}{\Longleftrightarrow} & & #2 \overset{#3}{=}& \ #4 & & \\
		\else\stackrel{#1}{\Longleftrightarrow} & & #2 \overset{#3}{=}& \ #4 & & |\ #5 \\ \fi
	\fi}

% Ausgemustert
%
% - Erhöht die Anzahl der möglichen Tabs
% \makeatletter\countdef\@maxtab=30\makeatother
% - Machteinen etwas größeren Punkt in der mitte (wurde für den Clock-Hand-Algorythmus verwendet)
% \newcommand*{\lc}{\raisebox{-.35ex}{\scalebox{1.6}{$\cdot$}}}