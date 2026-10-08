证明\(\boldsymbol{\hat{\beta}}_{\text{Ridge}}=\frac{1}{1+\lambda}\hat{\boldsymbol{\beta}}_{\text{OLS}}\)

因为：

\(\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}\)

\(\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\boldsymbol{X}^T\boldsymbol{X}+\lambda \boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}\)

因为正交设计 \(\boldsymbol{X}^T\boldsymbol{X}=\boldsymbol{I}\)：

\(\begin{aligned} \hat{\boldsymbol{\beta}}_{\text{Ridge}} &= (\boldsymbol{I}+\lambda \boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y} \\ &=\big((1+\lambda)\boldsymbol{I}\big)^{-1}\boldsymbol{X}^T\boldsymbol{y} \\ &=\frac{1}{1+\lambda}\boldsymbol{I}^{-1}\boldsymbol{X}^T\boldsymbol{y} \\ &=\frac{1}{1+\lambda}\boldsymbol{X}^T\boldsymbol{y} \end{aligned}\)

而 OLS 在正交条件下：

\(\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}= \boldsymbol{I}^{-1}\boldsymbol{X}^T\boldsymbol{y}= \boldsymbol{X}^T\boldsymbol{y}\)

把\(\hat{\boldsymbol{\beta}}_{\text{OLS}}=\boldsymbol{X}^T\boldsymbol{y}\)代入岭回归结果：

\(\boldsymbol{\hat{\beta}}_{\text{Ridge}}=\frac{1}{1+\lambda}\hat{\boldsymbol{\beta}}_{\text{OLS}}\)