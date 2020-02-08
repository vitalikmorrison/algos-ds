"""
/******************************************************************************

                            Online Java Compiler.
                Code, Compile, Run and Debug java program online.
Write your code in this editor and press "Run" button to execute it.

*******************************************************************************/
public class Main
{
	public static void main(String[] args) {
		System.out.println("Hello World");
		Fraction f1 = new Fraction (1 ,2);
        Fraction f2 = new Fraction (2 ,3);

        System.out.println(f1.add(f2));
        System.out.println(f1.add(1));
	}
}

class Fraction extends Number implements Comparable<Fraction>
{
    private Integer numerator;
    private Integer denominator;

    public Fraction(Integer num, Integer den) {
        this.numerator = num;
        this.denominator = den;
    }
    public Fraction(Integer num) {
        this.numerator = num;
        this.denominator = 1;
    }

    public Integer getNumerator(){
        return numerator;
    }

    public Integer getDenominator(){
        return denominator;
    }

    public void setNumerator(Integer num) {
        this.numerator = num;
    }

    public void setDenominator(Integer denominator) {
        this.denominator = denominator;
    }

    public Fraction add(Fraction other){
        Integer newNum, newDen, common;
        newNum = other.getDenominator()*this.numerator + this.denominator*other.getNumerator();
        newDen = other.getDenominator()*this.denominator;
        common = gcd(newNum, newDen);
        return new Fraction(newNum/common, newDen/common);

    }

    public Fraction add(Integer other) {
        return add(new Fraction(other));
    }

    private static Integer gcd(Integer m, Integer n) {
        while(m%n!=0) {
            Integer oldm = m;
            Integer oldn = n;

            m = oldn;
            n = oldm%oldn;
        }
        return n;
    }
    public String toString() {
        return numerator.toString() + "/" + denominator.toString();
    }

    public boolean equals(Fraction other){
        Integer num1 = this.numerator * other.getDenominator();
        Integer num2 = this.denominator * other.getNumerator();
        if (num2 == num1)
            return true;
        else
            return false;
    }
    public double doubleValue() {
        return numerator.doubleValue() / denominator.doubleValue();
    }

    public int intValue() {
        return numerator.intValue() / denominator.intValue();
    }
    public long longValue() {
        return numerator.longValue() / denominator.longValue();
    }
    public float floatValue() {
        return numerator.floatValue() / denominator.floatValue();
    }
    public int compareTo(Fraction other) {
        Integer num1 = this.numerator * other.getDenominator();
        Integer num2 = this.denominator * other.getNumerator();
        return num1 - num2;
    }

}

"""
