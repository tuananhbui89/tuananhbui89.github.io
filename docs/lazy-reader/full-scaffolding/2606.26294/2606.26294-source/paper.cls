\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{paper}[2026/06/03 v1.0 Modular NeurIPS/MLSys paper class]

\providecommand\FormatOverride{neurips}
\providecommand\LogosOverride{camlsys-cambridge}
\providecommand\TitleBgOverride{camlsys}
\edef\paper@format{\FormatOverride}
\edef\paper@logos{\LogosOverride}
\edef\paper@titlebg{\TitleBgOverride}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass{article}

\input{preamble/format.tex}

\endinput
