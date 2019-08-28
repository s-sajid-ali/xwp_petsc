for i in {1..100}
do
	SECONDS=0
		mpirun -np 8 ./ex_modify -prop_steps $i -ts_type cn -pc_type gamg -log_view &> out
	ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
	echo "$ELAPSED" >> "times_petsc"
	cat out | grep Main\ Stage: >> time_2
	python petsc_eff_print.py >> eff_petsc
done

