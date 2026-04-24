MAIN = main
PDF = $(MAIN).pdf

all: $(PDF)
	open -a Preview $(PDF)

$(PDF): $(MAIN).tex
	pdflatex $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.toc *.pdf
